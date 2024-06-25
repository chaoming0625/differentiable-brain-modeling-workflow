# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import os.path
import platform
import time
from typing import Any, Optional, Mapping, Dict, Callable, Union
from typing import Tuple

import matplotlib

from _arg_utls import MyArgumentParser

if platform.system().startswith('Linux'):
  matplotlib.use('agg')

parser = MyArgumentParser()

# Learning parameters
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
parser.add_argument("--epochs", type=int, default=10000, help="Number of training epochs.")
parser.add_argument("--opt", type=str, default='adam', help="Number of training epochs.")

# dataset
parser.add_argument("--batch_size", type=int, default=128, help="")
parser.add_argument("--warmup", type=float, default=0., help="The ratio for network simulation.")
parser.add_argument("--num_workers", type=int, default=4, help="")

# model
parser.add_argument("--dt", type=float, default=1., help="")
parser.add_argument("--neuron", type=str, default='gif', choices=['gif', 'alif'], help="")
parser.add_argument("--n_rec", type=int, default=200, help="")
parser.add_argument("--w_ei_ratio", type=float, default=4., help="")
parser.add_argument("--ff_scale", type=float, default=1., help="")
parser.add_argument("--rec_scale", type=float, default=0.5, help="")
parser.add_argument("--beta", type=float, default=1.0, help="")
parser.add_argument("--tau_a", type=float, default=1000., help="")
parser.add_argument("--tau_neu", type=float, default=100., help="")
parser.add_argument("--tau_syn", type=float, default=10., help="")
parser.add_argument("--tau_out", type=float, default=10., help="")
parser.add_argument("--conn_method", type=str, default='dense', help="")

# training parameters
parser.add_argument("--mode", type=str, default='train', help="")

# regularization parameters
parser.add_argument("--weight_L1", type=float, default=0.0, help="The weight L1 regularization.")
parser.add_argument("--weight_L2", type=float, default=0.0, help="The weight L2 regularization.")
gargs = parser.parse_args()

import brainunit as bu
import brainstate as bst
import brainpy as bp
import brainpy_datasets as bd
import brainscale as bnn
import braintools as bts
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import orbax.checkpoint
from jax.experimental.sparse.csr import csr_matvec_p, csr_matmat_p
from torch.utils.data import DataLoader, IterableDataset

bst.environ.set(
  dt=gargs.dt,
  mode=bst.mixin.JointMode(bst.mixin.Training(), bst.mixin.Batching())
)


class Checkpointer(orbax.checkpoint.CheckpointManager):
  def __init__(
      self,
      directory: str,
      max_to_keep: Optional[int] = None,
      save_interval_steps: int = 1,
      metadata: Optional[Mapping[str, Any]] = None,
  ):
    options = orbax.checkpoint.CheckpointManagerOptions(
      max_to_keep=max_to_keep,
      save_interval_steps=save_interval_steps,
      create=True
    )
    super().__init__(os.path.abspath(directory), options=options, metadata=metadata)

  def save(
      self,
      args: Any,
      step: int,
      metrics: Optional[Any] = None,
      force: Optional[bool] = False,
      **kwargs
  ):
    r = super().save(
      step,
      metrics=metrics,
      force=force,
      args=orbax.checkpoint.args.StandardSave(args)
    )
    return r

  def restore(
      self,
      args: Any = None,
      step: int = None,
      items: Any = None,
      **kwargs
  ):
    self.wait_until_finished()
    step = self.latest_step() if step is None else step
    if args is not None:
      tree = jax.tree_util.tree_map(orbax.checkpoint.utils.to_shape_dtype_struct, args)
      args = orbax.checkpoint.args.StandardRestore(tree)

    return super().restore(step, items=items, args=args)


class TaskData(IterableDataset):
  def __init__(self, task: bd.cognitive.CognitiveTask):
    self.task = task

  def __iter__(self):
    while True:
      yield self.task.sample_a_trial(0)[:2]


class TaskLoader(DataLoader):
  def __init__(self, dataset: bd.cognitive.CognitiveTask, *args, **kwargs):
    assert isinstance(dataset, bd.cognitive.CognitiveTask)
    super().__init__(TaskData(dataset), *args, **kwargs)


class SNNNet(bst.Module):
  def save_state(self, **kwargs) -> Dict:
    raise NotImplementedError

  def load_state(self, state_dict: Dict, **kwargs):
    raise NotImplementedError

  def vis_data(self) -> Dict:
    raise NotImplementedError

  def visualize(self, inputs, n2show: int = 5, filename: str = None):
    n_seq = inputs.shape[0]
    indices = np.arange(n_seq)
    batch_size = inputs.shape[1]
    bst.init_states(self, batch_size)

    def step(i, inp):
      with bst.environ.context(i=i, t=i * bst.environ.get_dt()):
        self.update(inp)
      return self.vis_data()

    res = bst.transform.for_loop(step, indices, inputs, pbar=bst.transform.ProgressBar(10))

    fig, gs = bp.visualize.get_figure(4, n2show, 3., 4.5)
    for i in range(n2show):
      # input spikes
      bp.visualize.raster_plot(indices, inputs[:, i], ax=fig.add_subplot(gs[0, i]), xlim=(0, n_seq))
      # recurrent spikes
      bp.visualize.raster_plot(indices, res['rec_spk'][:, i], ax=fig.add_subplot(gs[1, i]), xlim=(0, n_seq))
      # recurrent membrane potentials
      ax = fig.add_subplot(gs[2, i])
      ax.plot(indices, res['rec_mem'][:, i])
      # output potentials
      ax = fig.add_subplot(gs[3, i])
      ax.plot(indices, res['out'][:, i])

    if filename is None:
      plt.show()
      plt.close()
    else:
      plt.savefig(filename)
      plt.close()


class GIF(bst.nn.Neuron):
  def __init__(
      self, size, V_rest=0., V_th_inf=1., tau=20., tau_a=50., beta=1.,
      V_initializer: Callable = bst.init.ZeroInit(),
      I2_initializer: Callable = bst.init.ZeroInit(),
      spike_fun: Callable = bst.surrogate.ReluGrad(),
      spk_reset: str = 'soft',
      keep_size: bool = False,
      name: str = None,
      mode: bst.mixin.Mode = None,
  ):
    super().__init__(size, keep_size=keep_size, name=name, mode=mode, spk_fun=spike_fun, spk_reset=spk_reset)

    # params
    self.V_rest = bst.init.param(V_rest, self.varshape, allow_none=False)
    self.V_th_inf = bst.init.param(V_th_inf, self.varshape, allow_none=False)
    self.tau = bst.init.param(tau, self.varshape, allow_none=False)
    self.tau_I2 = bst.init.param(tau_a, self.varshape, allow_none=False)
    self.beta = bst.init.param(beta, self.varshape, allow_none=False)

    # initializers
    self._V_initializer = V_initializer
    self._I_initializer = I2_initializer

  def init_state(self, batch_size=None):
    self.V = bnn.ETraceVar(bst.init.param(self._V_initializer, self.varshape, batch_size))
    self.I_adp = bnn.ETraceVar(bst.init.param(self._I_initializer, self.varshape, batch_size))

  def dI2(self, I2, t):
    return - I2 / self.tau_I2

  def dV(self, V, t, I_ext):
    I_ext = self.sum_current_inputs(V, init=I_ext)
    return (- V + self.V_rest + I_ext) / self.tau

  def update(self, x=0.):
    t = bst.environ.get('t')
    last_spk = self.get_spike()
    last_V = self.V.value - self.V_th_inf * last_spk
    last_I2 = self.I_adp.value - self.beta * last_spk
    I2 = bst.nn.exp_euler_step(self.dI2, last_I2, t)
    V = bst.nn.exp_euler_step(self.dV, last_V, t, I_ext=(x + I2))
    V += self.sum_delta_inputs()
    self.I_adp.value = I2
    self.V.value = V
    return self.get_spike(V)

  def get_spike(self, V=None):
    V = self.V.value if V is None else V
    return self.spk_fun((V - self.V_th_inf) / self.V_th_inf)


class SNNCobaNet(SNNNet):
  def __init__(
      self, n_in, n_rec, n_out, tau_neu=10., tau_a=100., beta=1., tau_syn=10., tau_out=10.,
      ff_scale=1., rec_scale=1., E_exc=3., E_inh=-3., w_ei_ratio: float = 10.,
      conn_method: str = 'dense'
  ):
    super().__init__()

    self.n_exc = int(n_rec * 0.8)
    self.n_inh = n_rec - self.n_exc

    # neurons
    self.pop = GIF(n_rec, tau=tau_neu, tau_a=bst.init.Uniform(tau_a * 0.5, tau_a * 1.5), beta=beta)
    ff_init = bst.init.KaimingNormal(scale=ff_scale)
    # feedforward
    self.ff2r = bst.nn.HalfProjAlignPostMg(
      comm=bnn.SignedWLinear(n_in, n_rec, w_init=ff_init),
      syn=bnn.Expon.delayed(size=n_rec, tau=tau_syn),
      out=bst.nn.COBA.delayed(E=E_exc),
      post=self.pop
    )
    # recurrent
    if conn_method == 'dense':
      inh_init = bst.init.KaimingNormal(scale=rec_scale * w_ei_ratio)
      exc_init = bst.init.KaimingNormal(scale=rec_scale)
      inh2r_conn = bnn.SignedWLinear(self.n_inh, n_rec, w_init=inh_init)
      exc2r_conn = bnn.SignedWLinear(self.n_exc, n_rec, w_init=exc_init)
    elif conn_method == 'rand':
      inh2r_conn = FixedProbCSR(0.1, self.n_inh, n_rec, rec_scale * w_ei_ratio, seed=123)
      exc2r_conn = FixedProbCSR(0.1, self.n_exc, n_rec, rec_scale, seed=1234)
    elif conn_method == 'gaussian':
      inh2r_conn = FixedProbCSR(0.1, self.n_inh, n_rec, rec_scale * w_ei_ratio, seed=123)
      exc2r_conn = GaussianCSR(20., self.n_exc, n_rec, rec_scale, seed=1234)
    else:
      raise ValueError(f'Unknown connection method: {conn_method}')

    self.inh2r = bst.nn.HalfProjAlignPostMg(
      comm=inh2r_conn,
      syn=bnn.Expon.delayed(size=n_rec, tau=tau_syn),
      out=bst.nn.COBA.delayed(E=E_inh),
      post=self.pop
    )
    self.exc2r = bst.nn.HalfProjAlignPostMg(
      comm=exc2r_conn,
      syn=bnn.Expon.delayed(size=n_rec, tau=tau_syn),
      out=bst.nn.COBA.delayed(E=E_exc),
      post=self.pop
    )
    # output
    self.out = bnn.LeakyRateReadout(n_rec, n_out, tau=tau_out)

  def update(self, spk):
    e_sps, i_sps = jnp.split(self.pop.get_spike(), [self.n_exc], axis=-1)
    self.ff2r(spk)
    self.exc2r(e_sps)
    self.inh2r(i_sps)
    return self.out(self.pop())

  def save_state(self, **kwargs) -> Dict:
    return {'ff2r.weight': self.ff2r.comm.weight_op.value,
            'exc2r.weight': self.exc2r.comm.weight_op.value,
            'inh2r.weight': self.inh2r.comm.weight_op.value,
            'out.weight': self.out.weight_op.value}

  def load_state(self, state_dict: Dict, **kwargs):
    self.ff2r.comm.weight_op.value = state_dict['ff2r.weight']
    self.exc2r.comm.weight_op.value = state_dict['exc2r.weight']
    self.inh2r.comm.weight_op.value = state_dict['inh2r.weight']
    self.out.weight_op.value = state_dict['out.weight']

  def vis_data(self):
    n_rec = self.pop.num
    return {'rec_spk': self.pop.get_spike(),
            'rec_mem': self.pop.V.value[:, np.arange(0, n_rec, n_rec // 10)],
            'out': self.out.r.value, }


def csr_matvec(
    data: bst.typing.ArrayLike,
    indices: bst.typing.ArrayLike,
    indptr: bst.typing.ArrayLike,
    v: bst.typing.ArrayLike,
    *,
    shape,
    transpose=False
) -> jax.Array:
  """Product of CSR sparse matrix and a dense vector.

  Args:
    data : array of shape ``(nse,)``.
    indices : array of shape ``(nse,)``
    indptr : array of shape ``(shape[0] + 1,)`` and dtype ``indices.dtype``
    v : array of shape ``(shape[0] if transpose else shape[1],)``
      and dtype ``data.dtype``
    shape : length-2 tuple representing the matrix shape
    transpose : boolean specifying whether to transpose the sparse matrix
      before computing.

  Returns:
    y : array of shape ``(shape[1] if transpose else shape[0],)`` representing
      the matrix vector product.
  """
  return csr_matvec_p.bind(data, indices, indptr, v, shape=shape, transpose=transpose)


def csr_matmat(
    data: bst.typing.ArrayLike,
    indices: bst.typing.ArrayLike,
    indptr: bst.typing.ArrayLike,
    B: bst.typing.ArrayLike,
    *,
    shape,
    transpose: bool = False
) -> jax.Array:
  """Product of CSR sparse matrix and a dense matrix.

  Args:
    data : array of shape ``(nse,)``.
    indices : array of shape ``(nse,)``
    indptr : array of shape ``(shape[0] + 1,)`` and dtype ``indices.dtype``
    B : array of shape ``(shape[0] if transpose else shape[1], cols)`` and
      dtype ``data.dtype``
    shape : length-2 tuple representing the matrix shape
    transpose : boolean specifying whether to transpose the sparse matrix
      before computing.

  Returns:
    C : array of shape ``(shape[1] if transpose else shape[0], cols)``
      representing the matrix-matrix product.
  """
  return csr_matmat_p.bind(data, indices, indptr, B, shape=shape, transpose=transpose)


class CSRLayer(bst.nn.DnnLayer):
  def __init__(
      self,
      conn: bp.conn.TwoEndConnector,
      weight: Union[bst.typing.ArrayLike, Callable],
      w_sign: float | None = None,
      transpose: bool = True,
      mode: bst.environ.Mode = None
  ):
    super().__init__(mode=mode)

    assert isinstance(conn, bp.conn.TwoEndConnector)
    self.conn = conn
    self.transpose = transpose
    self.w_sign = w_sign

    # connection
    self.indices, self.indptr = self.conn.require('csr')

    # weight
    weight = bst.init.param(weight, (self.indices.size,), allow_scalar=True)
    if self.mode.has(bst.mixin.Training):
      if np.isscalar(weight):
        weight = jnp.full(self.indices.size, weight, dtype=bst.environ.dftype())
      weight = bnn.ETraceParamOp(weight, self._op)
    self.weight_op = weight

  def to_dense_conn(self):
    data = self.weight_op.value if isinstance(self.weight_op, bnn.ETraceParamOp) else self.weight_op
    return bp.math.sparse.csr_to_dense(data, self.indices, self.indptr, shape=(self.conn.pre_num, self.conn.post_num))

  def update(self, x):
    if self.mode.has(bst.mixin.Training):
      return self.weight_op.execute(x)
    return self._op(x, self.weight_op)

  def _op(self, x, w):
    if self.w_sign is None:
      w = jnp.abs(w)
    else:
      w = jnp.abs(w) * self.w_sign

    if x.ndim == 1:
      # forward event-driven computation
      if jnp.isscalar(w):
        return bp.math.event.csrmv(
          w,
          self.indices,
          self.indptr, x,
          shape=(self.conn.pre_num, self.conn.post_num),
          transpose=self.transpose
        )
      else:
        return bp.math.event.csrmv(
          w,
          self.indices,
          self.indptr, x,
          shape=(self.conn.pre_num, self.conn.post_num),
          transpose=self.transpose
        )
    else:
      shapes = x.shape[:-1]
      x = bu.math.flatten(x, end_dim=-2)
      y = csr_matmat(
        w,
        self.indices,
        self.indptr,
        x.T,
        shape=(self.conn.pre_num, self.conn.post_num),
        transpose=self.transpose
      )
      y = y.T
      return jnp.reshape(y, shapes + (y.shape[-1],))


class FixedProbCSR(CSRLayer):
  def __init__(
      self,
      prob,
      n_pre,
      n_post,
      scale,
      w_sign: float | None = None,
      transpose: bool = True,
      mode: bst.environ.Mode = None,
      seed=None
  ):
    self.n_pre = n_pre
    self.scale = scale
    self.prob = prob
    conn = bp.conn.FixedProb(prob, pre=n_pre, post=n_post, seed=seed)

    def init(shape):
      variance = self.scale / (self.n_pre * prob)
      stddev = jnp.sqrt(variance) / .87962566103423978
      return bst.random.truncated_normal(-2, 2, shape, dtype=bst.environ.dftype()) * stddev

    super().__init__(conn, init, w_sign=w_sign, transpose=transpose, mode=mode)


class GaussianCSR(CSRLayer):
  def __init__(
      self,
      sigma,
      n_pre,
      n_post,
      scale,
      w_sign: float | None = None,
      transpose: bool = True,
      mode: bst.environ.Mode = None,
      seed=None
  ):
    self.n_pre = n_pre
    self.scale = scale
    conn = bp.conn.GaussianProb(sigma, pre=n_pre, post=n_post, seed=seed)

    def init(shape):
      variance = self.scale / self.n_pre
      stddev = jnp.sqrt(variance) / .87962566103423978
      return bst.random.truncated_normal(-2, 2, shape, dtype=bst.environ.dftype()) * stddev

    super().__init__(conn, init, w_sign=w_sign, transpose=transpose, mode=mode)


class Trainer:
  def __init__(
      self,
      target_net: SNNNet,
      optimizer: bst.optim.Optimizer,
      loader: bd.cognitive.TaskLoader,
      args: bst.util.DotDict,
      filepath: str | None = None
  ):
    # the network
    self.target_net = target_net

    # the dataset
    self.loader = loader

    # parameters
    self.args = args
    self.filepath = filepath
    self.checkpointer = None
    if filepath is not None:
      self.checkpointer = Checkpointer(filepath, max_to_keep=10)

    # optimizer
    weights = self.target_net.states().subset(bst.ParamState)
    print(weights)
    self.optimizer = optimizer
    self.optimizer.register_trainable_weights(weights)

  def print(self, msg, file=None):
    if file is not None:
      print(msg, file=file)
    print(msg)

  def _loss(self, out, target):
    # MSE loss
    mse_loss = bts.metric.softmax_cross_entropy_with_integer_labels(out, target).mean()
    # L1 regularization loss
    l1_loss = 0.
    if self.args.weight_L1 != 0.:
      leaves = self.target_net.states().subset(bst.ParamState).to_dict_values()
      for leaf in leaves:
        l1_loss += self.args.weight_L1 * jnp.sum(jnp.abs(leaf))
    return mse_loss, l1_loss

  def _acc(self, outs, target):
    pred = jnp.argmax(jnp.sum(outs, 0), 1)  # [T, B, N] -> [B, N] -> [B]
    acc = jnp.asarray(pred == target, dtype=bst.environ.dftype()).mean()
    return acc

  @bst.transform.jit(static_argnums=(0,))
  def bptt_train(self, inputs, targets) -> Tuple:
    inputs = jnp.asarray(inputs, dtype=bst.environ.dftype())
    indices = jnp.arange(inputs.shape[0])
    bst.init_states(self.target_net, inputs.shape[1])
    weights = self.target_net.states().subset(bst.ParamState)
    warmup = self.args.warmup + inputs.shape[0] if self.args.warmup < 0 else self.args.warmup
    n_sim = int(warmup) if warmup > 0 else 0

    def _step_run(i, inp):
      with bst.environ.context(i=i, t=i * bst.environ.get_dt()):
        out = self.target_net(inp)
      return self._loss(out, targets), out

    def _bptt_grad():
      (mse_losses, l1_losses), outs = bst.transform.for_loop(_step_run, indices, inputs)
      mse_losses = mse_losses[n_sim:].mean()
      l1_losses = l1_losses[n_sim:].mean()
      acc = self._acc(outs[n_sim:], targets)
      return mse_losses + l1_losses, (mse_losses, l1_losses, acc)

    f_grad = bst.transform.grad(_bptt_grad, grad_vars=weights, has_aux=True, return_value=True)
    grads, loss, (mse_losses, l1_losses, acc) = f_grad()
    self.optimizer.update(grads)
    return mse_losses, l1_losses, acc

  def f_sim(self):
    inputs, outputs = next(iter(self.loader))
    inputs = jnp.asarray(inputs, dtype=bst.environ.dftype()).transpose(1, 0, 2)
    self.target_net.visualize(inputs)

  def f_train(self):
    file = None
    if self.filepath is not None:
      if not os.path.exists(self.filepath):
        os.makedirs(self.filepath)
      file = open(f'{self.filepath}/loss.txt', 'w')
    self.print(self.args, file=file)

    acc_max = 0.
    t0 = time.time()
    for bar_idx, (inputs, outputs) in enumerate(self.loader):
      if bar_idx > gargs.epochs:
        break

      inputs = jnp.asarray(inputs, dtype=bst.environ.dftype()).transpose(1, 0, 2)
      outputs = jnp.asarray(outputs, dtype=bst.environ.ditype())
      mse_ls, l1_ls, acc = self.bptt_train(inputs, outputs)
      self.optimizer.lr.step_epoch()
      desc = (f'Batch {bar_idx:2d}, '
              f'CE={float(mse_ls):.8f}, '
              f'L1={float(l1_ls):.6f}, '
              f'acc={float(acc):.6f}, '
              f'time={time.time() - t0:.2f} s')
      self.print(desc, file=file)

      if acc > acc_max and self.checkpointer is not None:
        acc_max = acc
        weights = jax.tree.map(np.asarray, self.target_net.save_state())
        self.checkpointer.save(weights, step=bar_idx)
        self.target_net.visualize(inputs, filename=f'{self.filepath}/{bar_idx}/train-results-{bar_idx}.png')

      t0 = time.time()
      if acc_max > 0.99:
        break
    if file is not None:
      file.close()


def training():
  # filepath
  now = time.strftime('%Y-%m-%d %H-%M-%S', time.localtime(int(round(time.time() * 1000)) / 1000))
  if gargs.method == 'bptt':
    filepath = f'results/task-training/{gargs.method}-{gargs.conn_method}-{now}'
  else:
    filepath = (f'results/task-training/'
                f'{gargs.method}-{gargs.diag_jacobian}-{gargs.diag_normalize}-{gargs.conn_method}-{now}')
  # filepath = None

  # data
  task = bd.cognitive.EvidenceAccumulation(dt=bst.environ.get_dt(), mode='spiking', )
  gargs.warmup = -(task.t_recall / bst.environ.get_dt())
  loader = TaskLoader(task, batch_size=gargs.batch_size, num_workers=gargs.num_workers)

  # network
  net = SNNCobaNet(
    task.num_inputs,
    gargs.n_rec,
    task.num_outputs,
    beta=gargs.beta,
    tau_a=gargs.tau_a,
    tau_neu=gargs.tau_neu,
    tau_syn=gargs.tau_syn,
    tau_out=gargs.tau_out,
    ff_scale=gargs.ff_scale,
    rec_scale=gargs.rec_scale,
    w_ei_ratio=gargs.w_ei_ratio,
    conn_method=gargs.conn_method,
  )

  # optimizer
  if gargs.opt == 'adam':
    opt = bst.optim.Adam(lr=gargs.lr)
  elif gargs.opt == 'sgd':
    opt = bst.optim.SGD(lr=gargs.lr)
  else:
    raise ValueError

  # trainer
  trainer = Trainer(net, opt, loader, gargs, filepath=filepath)

  if gargs.mode == 'sim':
    trainer.f_sim()
  else:
    trainer.f_train()


def verification():
  import seaborn as sns

  filepath = r'results\task-training\bptt-ec-snn_coba-gif-dense-2024-05-30 13-54-18'

  checkpointer = Checkpointer(filepath, max_to_keep=10)

  def visualize_activity(self, inputs, n2show: int = 4, filename: str = None):
    n_seq = inputs.shape[0]
    indices = np.arange(n_seq)
    batch_size = inputs.shape[1]
    bst.init_states(self, batch_size)

    def step(i, inp):
      with bst.environ.context(i=i, t=i * bst.environ.get_dt()):
        self.update(inp)
        n_exc = int(self.pop.num * 0.8)
        n_inh = self.pop.num - n_exc
        exc_indices = np.arange(0, n_exc, n_exc // 5)
        inh_indices = np.arange(0, n_inh, n_inh // 5) + n_exc
        return {'rec_spk': self.pop.get_spike(),
                'exc_mem': self.pop.V.value[:, exc_indices],
                'inh_mem': self.pop.V.value[:, inh_indices],
                'out': self.out.r.value, }

    res = bst.transform.for_loop(step, indices, inputs, pbar=bst.transform.ProgressBar(10))

    fig, gs = bp.visualize.get_figure(5, n2show, 2.0, 4.)
    for i in range(n2show):
      # input spikes
      bp.visualize.raster_plot(indices, inputs[:, i], ax=fig.add_subplot(gs[0, i]), xlim=(0, n_seq))
      # recurrent spikes
      bp.visualize.raster_plot(indices, res['rec_spk'][:, i], ax=fig.add_subplot(gs[1, i]), xlim=(0, n_seq))
      # recurrent membrane potentials
      ax = fig.add_subplot(gs[2, i])
      ax.plot(indices, res['exc_mem'][:, i])
      # recurrent membrane potentials
      ax = fig.add_subplot(gs[3, i])
      ax.plot(indices, res['inh_mem'][:, i])
      # output potentials
      ax = fig.add_subplot(gs[4, i])
      ax.plot(indices, res['out'][:, i])

    if filename is None:
      plt.show()
      plt.close()
    else:
      plt.savefig(filename)
      plt.close()

  def visualize_weights(self, show=True):
    if gargs.conn_method == 'dense':
      weights = jnp.abs(jnp.concat([self.exc2r.comm.weight_op.value, self.inh2r.comm.weight_op.value], axis=0))
    else:
      weights = jnp.abs(jnp.concat([self.exc2r.comm.to_dense_conn(), self.inh2r.comm.to_dense_conn()], axis=0))
    weights = np.ma.array(weights, mask=weights == 0)

    fig, gs = bp.visualize.get_figure(1, 1, 5., 5.)
    ax = fig.add_subplot(gs[0, 0])
    # pcolormesh = plt.pcolormesh(weights, cmap='Purples')
    # pcolormesh = plt.pcolormesh(weights, cmap='Reds')
    # pcolormesh = plt.pcolormesh(weights, cmap='seismic')
    pcolormesh = plt.pcolormesh(weights, cmap='cool', vmin=0.0, vmax=1.5)
    cmap = pcolormesh.cmap  # Get the colormap
    cmap.set_bad('white', 1.)  # Set white for NaN values with full alpha
    plt.colorbar(pcolormesh)
    plt.xlabel('To neurons')
    plt.ylabel('From neurons')
    plt.title('Network connectivity')
    if show:
      plt.show()

  def plot_weight_dists(self, show=True):
    exc_weight = np.abs(self.exc2r.comm.weight_op.value).flatten()
    inh_weight = np.abs(self.inh2r.comm.weight_op.value).flatten()

    fig, gs = bp.visualize.get_figure(1, 2, 3, 3.)
    ax = fig.add_subplot(gs[0, 0])
    bin_res = plt.hist(exc_weight, bins=100, color='blue', alpha=0.7, density=True)
    plt.title('Excitatory weights')
    sns.kdeplot(exc_weight, thresh=0.01)
    plt.xlim(0., bin_res[1].max())

    ax = fig.add_subplot(gs[0, 1])
    bin_res = plt.hist(inh_weight, bins=100, color='blue', alpha=0.7, density=True)
    sns.kdeplot(inh_weight, thresh=0.01)
    plt.title('Inhibitory weights')

    if show:
      plt.show()

  global gargs
  with open(os.path.join(filepath, 'loss.txt'), 'r') as f:
    line = f.readline().strip().replace('Namespace', 'dict')
    gargs = bst.util.DotDict(eval(line))
    print(gargs)

  bst.environ.set(
    dt=gargs.dt,
    mode=bst.mixin.JointMode(bst.mixin.Training(), bst.mixin.Batching())
  )
  task = bd.cognitive.EvidenceAccumulation(dt=bst.environ.get_dt(), mode='spiking')
  loader = TaskLoader(task, batch_size=gargs.batch_size, num_workers=gargs.num_workers)
  gargs.warmup = -(task.t_recall / bst.environ.get_dt())
  net = SNNCobaNet(task.num_inputs,
                   gargs.n_rec,
                   task.num_outputs,
                   beta=gargs.beta,
                   tau_a=gargs.tau_a,
                   tau_neu=gargs.tau_neu,
                   tau_syn=gargs.tau_syn,
                   tau_out=gargs.tau_out,
                   ff_scale=gargs.ff_scale,
                   rec_scale=gargs.rec_scale,
                   w_ei_ratio=gargs.w_ei_ratio,
                   conn_method=gargs.conn_method, )

  # visualize_weights(net, show=False)
  plot_weight_dists(net, show=False)
  weight_data = checkpointer.restore(net.save_state())
  net.load_state(weight_data)
  # visualize_weights(net)
  plot_weight_dists(net)

  inputs, _ = next(iter(loader))
  inputs = jnp.asarray(inputs, dtype=bst.environ.dftype()).transpose(1, 0, 2)
  visualize_activity(net, inputs)


if __name__ == '__main__':
  pass
  training()

# python task-coba-ei-rsnn.py --method bptt
# python task-coba-ei-rsnn.py --method diag
# python task-coba-ei-rsnn.py --method expsm_diag --etrace_decay 0.98
