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


from functools import partial
from typing import Union, Callable, Optional

import brainstate as bst
import jax
import matplotlib.pyplot as plt
import numpy as np
import braintools as bts

from _utils import gamma_factor, NevergradOptimizer, ScipyOptimizer, SkoptOptimizer

bst.environ.set(dt=0.1)

inp_traces = [
  bts.input.constant_input([(2., 200.), (0., 800.)])[0],
  bts.input.constant_input([(1.5, 500.), (0., 500.)])[0],
  bts.input.constant_input([(1.5, 100.), (0, 500.), (0.5, 100.), (1., 100.), (1.5, 100.), (0., 100.)])[0],
  bts.input.constant_input([(1.5, 20.), (0., 180.), (-1.5, 20.), (0., 20.), (1.5, 20.), (0., 140.), (0., 600.)])[0],
  bts.input.constant_input([(0, 50.), (-3.5, 750.), (0., 200.)])[0]
]
inp_traces = np.asarray(inp_traces)


def visualize(currents, voltages, gl, g_na, g_kd, vth):
  # currents: [T, B]
  # voltages: [T, B]
  simulated_vs = simulate_model(currents, gl, g_na, g_kd, vth)[1]
  currents = np.asarray(currents)
  voltages = np.asarray(voltages)
  simulated_vs = np.asarray(simulated_vs)

  fig, gs = bts.visualize.get_figure(2, simulated_vs.shape[1], 3, 4.5)
  for i in range(simulated_vs.shape[1]):
    ax = fig.add_subplot(gs[0, i])
    ax.plot(voltages[:, i], label='target')
    ax.plot(simulated_vs[:, i], label='simulated')
    plt.legend()
    ax = plt.subplot(gs[1, i])
    ax.plot(currents[:, i])
  plt.show()


class GIF(bst.nn.Neuron):
  def __init__(
      self,
      size: bst.typing.Size,
      keep_size: bool = False,
      mode: Optional[bst.mixin.Mode] = None,
      name: Optional[str] = None,
      spk_fun: Callable = bst.surrogate.ReluGrad(),
      spk_reset: str = 'soft',
      detach_spk: bool = False,

      # neuron parameters
      V_rest: Union[bst.typing.ArrayLike, Callable] = -70.,
      V_reset: Union[bst.typing.ArrayLike, Callable] = -70.,
      V_th_inf: Union[bst.typing.ArrayLike, Callable] = -50.,
      V_th_reset: Union[bst.typing.ArrayLike, Callable] = -60.,
      R: Union[bst.typing.ArrayLike, Callable] = 20.,
      tau: Union[bst.typing.ArrayLike, Callable] = 20.,
      a: Union[bst.typing.ArrayLike, Callable] = 0.,
      b: Union[bst.typing.ArrayLike, Callable] = 0.01,
      k1: Union[bst.typing.ArrayLike, Callable] = 0.2,
      k2: Union[bst.typing.ArrayLike, Callable] = 0.02,
      R1: Union[bst.typing.ArrayLike, Callable] = 0.,
      R2: Union[bst.typing.ArrayLike, Callable] = 1.,
      A1: Union[bst.typing.ArrayLike, Callable] = 0.,
      A2: Union[bst.typing.ArrayLike, Callable] = 0.,
      V_initializer: Union[Callable, bst.typing.ArrayLike] = bst.init.Constant(-70.),
      I1_initializer: Union[Callable, bst.typing.ArrayLike] = bst.init.Constant(0.),
      I2_initializer: Union[Callable, bst.typing.ArrayLike] = bst.init.Constant(0.),
      Vth_initializer: Union[Callable, bst.typing.ArrayLike] = bst.init.Constant(-50.),
  ):
    # initialization
    super().__init__(size,
                     name=name,
                     keep_size=keep_size,
                     mode=mode,
                     spk_fun=spk_fun,
                     detach_spk=detach_spk,
                     spk_reset=spk_reset)
    # parameters
    self.V_rest = bst.init.param(V_rest, self.varshape)
    self.V_reset = bst.init.param(V_reset, self.varshape)
    self.V_th_inf = bst.init.param(V_th_inf, self.varshape)
    self.V_th_reset = bst.init.param(V_th_reset, self.varshape)
    self.R = bst.init.param(R, self.varshape)
    self.a = bst.init.param(a, self.varshape)
    self.b = bst.init.param(b, self.varshape)
    self.k1 = bst.init.param(k1, self.varshape)
    self.k2 = bst.init.param(k2, self.varshape)
    self.R1 = bst.init.param(R1, self.varshape)
    self.R2 = bst.init.param(R2, self.varshape)
    self.A1 = bst.init.param(A1, self.varshape)
    self.A2 = bst.init.param(A2, self.varshape)
    self.tau = bst.init.param(tau, self.varshape)

    # initializers
    self._V_initializer = V_initializer
    self._I1_initializer = I1_initializer
    self._I2_initializer = I2_initializer
    self._Vth_initializer = Vth_initializer

  def dI1(self, I1, t):
    return - self.k1 * I1

  def dI2(self, I2, t):
    return - self.k2 * I2

  def dVth(self, V_th, t, V):
    return self.a * (V - self.V_rest) - self.b * (V_th - self.V_th_inf)

  def dV(self, V, t, I1, I2, I):
    I = self.sum_current_inputs(V, init=I)
    return (- (V - self.V_rest) + self.R * (I + I1 + I2)) / self.tau

  def init_state(self, batch_size=None, **kwargs):
    self.V = bst.ShortTermState(bst.init.param(self._V_initializer, self.varshape, batch_size))
    self.I1 = bst.ShortTermState(bst.init.param(self._I1_initializer, self.varshape, batch_size))
    self.I2 = bst.ShortTermState(bst.init.param(self._I2_initializer, self.varshape, batch_size))
    self.V_th = bst.ShortTermState(bst.init.param(self._Vth_initializer, self.varshape, batch_size))

  def get_spike(self, V, v_th):
    return self.spk_fun((V - v_th) / jax.numpy.abs(v_th))

  def update(self, x=0.):
    t = bst.environ.get('t')

    last_v = self.V.value
    last_I1 = self.I1.value
    last_I2 = self.I2.value
    last_V_th = self.V_th.value
    last_spike = self.get_spike(last_v, last_V_th)
    if self.detach_spk:
      last_spike = jax.lax.stop_gradient(last_spike)

    last_v = last_v - (last_v - self.V_reset) * last_spike
    last_I1 += last_spike * (self.R1 * last_I1 + self.A1 - last_I1)
    last_I2 += last_spike * (self.R2 * last_I2 + self.A2 - last_I2)

    # integrate membrane potential
    I1 = bst.nn.exp_euler_step(self.dI1, last_I1, t)
    I2 = bst.nn.exp_euler_step(self.dI2, last_I2, t)
    V_th = bst.nn.exp_euler_step(self.dVth, last_V_th, t, last_v)
    V = bst.nn.exp_euler_step(self.dV, last_v, t, I1, I2, x)
    V += self.sum_delta_inputs()

    # assign new values
    V_th = jax.numpy.maximum(self.V_th_reset, V_th)
    spike = self.get_spike(V, V_th)
    self.V.value = V
    self.I1.value = I1
    self.I2.value = I2
    self.V_th.value = V_th
    return spike


@bst.transform.jit
def simulate_model(current, a=0., b=0.01, k1=0.2, k2=0.02, ):
  assert current.ndim == 2  # [T, B]
  n_input = current.shape[1]
  neu = GIF(n_input, a=a, b=b, k1=k1, k2=k2)
  neu.init_state()

  def step_fun(i, inp):
    with bst.environ.context(i=i, t=bst.environ.get_dt() * i):
      spk = neu.update(inp)
    return spk, neu.V.value, neu.I1.value, neu.I2.value, neu.V_th.value

  indices = np.arange(current.shape[0])
  return bst.transform.for_loop(step_fun, indices, current)  # (T, B)


def compare_spikes(param, currents, target_spks):
  spks = simulate_model(currents, *param)[0]  # (T, B)
  losses = jax.vmap(partial(gamma_factor, dt=bst.environ.get_dt()), axis_size=1)(spks, target_spks)
  return losses.sum()


def compare_potentials(param, currents, target_potentials, n_point=10):
  vs = simulate_model(currents, *param)[1]  # (T, B)
  indices = np.arange(0, vs.shape[0], vs.shape[0] // n_point)
  losses = bts.metric.squared_error(vs[indices], target_potentials[indices])
  return losses.mean()


# inp_traces: [B, T]
# with jax.disable_jit():
target_hists = simulate_model(inp_traces.T, 0.005)


def visualize_fitting_trarget():
  # plt.style.use(['science', 'nature', 'notebook'])
  indices = np.arange(inp_traces.shape[1])

  # fig, gs = bp.visualize.get_figure(1, 1, 3.0, 4.5)
  # ax = fig.add_subplot(gs[0, 0])
  # ax.plot(indices, target_hists[1])
  # # plt.ylabel('Potential [mV]')
  # # plt.xlabel('Time [ms]')
  # plt.xticks([])
  # plt.yticks([])
  # plt.savefig('hh-simulate.eps', dpi=300, transparent=True)
  # plt.show()

  fig, gs = bts.visualize.get_figure(2, inp_traces.shape[0], 3.0, 4.5)
  for i in range(inp_traces.shape[0]):
    ax = fig.add_subplot(gs[0, i])
    ax.plot(indices, inp_traces[i])

    ax = fig.add_subplot(gs[1, i])
    ax.plot(indices, target_hists[1][:, i], label='V')
    ax.plot(indices, target_hists[-1][:, i], label='V_th')
    # plt.ylabel('Current [nA]')
    # plt.xlabel('Time [ms]')
    plt.legend()
    # plt.xticks([])
    # plt.yticks([])
  plt.show()


bounds = [np.asarray([0., 0.00, 0.1, 0.01]),
          np.asarray([0.1, 0.1, 1., 0.1])]


def fitting_by_gradient(fit_target='spike', n_sample=100):
  print(f"Method: L-BFGS-B, fit_target: {fit_target}, n_sample: {n_sample}")

  if fit_target == 'spike':
    fun = jax.jit(partial(compare_spikes, currents=inp_traces.T, target_spks=target_hists[0]))
  elif fit_target == 'potential':
    fun = jax.jit(lambda x: compare_potentials(x, inp_traces.T, target_hists[1], n_point=100))
  else:
    raise ValueError(f"Unknown fit target: {fit_target}")

  # opt = BFGSOptimizer(fun, bounds=bounds, bound_factor=100.)
  opt = ScipyOptimizer(fun, bounds=bounds)
  param = opt.minimize(num_sample=n_sample)

  print(param.x)
  print(param.fun)

  return param.x, param.fun


def fitting_by_others(fit_target='spike', method='skopt', n_sample=20):
  print(f"Method: {method}, fit_target: {fit_target}, n_sample: {n_sample}")

  if fit_target == 'spike':
    fun = jax.jit(partial(compare_spikes, currents=inp_traces.T, target_spks=target_hists[0]))
  elif fit_target == 'potential':
    fun = jax.jit(partial(compare_potentials, currents=inp_traces.T, target_potentials=target_hists[1], n_point=100))
  else:
    raise ValueError(f"Unknown fit target: {fit_target}")

  @jax.jit
  @jax.vmap
  def loss_with_multiple_run(a, b, k1, k2):
    return fun([a, b, k1, k2])

  if method in ['bayesian']:
    opt = SkoptOptimizer(
      loss_with_multiple_run,
      n_sample=n_sample,
      bounds=np.asarray(bounds).T,
    )
  else:
    opt = NevergradOptimizer(
      loss_with_multiple_run,
      n_sample=n_sample,
      bounds={'a': (bounds[0][0], bounds[1][0]),
              'b': (bounds[0][1], bounds[1][1]),
              'k1': (bounds[0][2], bounds[1][2]),
              'k2': (bounds[0][3], bounds[1][3])},
      use_nevergrad_recommendation=False,
      method=method,
    )

  opt.initialize()
  param = opt.minimize(5)
  loss = fun(param)

  print(param)
  print(loss)
  # visualize(inp_traces.T, target_hists[1], *param)

  return param, loss


def compare_different_fitting_v2(fit_target='spike'):
  def add_current(fig, gs, i_ax):
    for i in range(currents.shape[1]):
      ax = fig.add_subplot(gs[i_ax:i_ax + 2, i])
      ax.plot(times, currents[:, i])
      if i == 0:
        plt.ylabel('Current [nA]')
      plt.xticks([])
    return i_ax + 2

  def add_plot(params, fig, gs, ax_idx, label, xlabel=False):
    vs = simulate_model(currents, *params)[1]
    for i in range(currents.shape[1]):
      ax2 = fig.add_subplot(gs[ax_idx:ax_idx + 3, i])
      ax2.plot(times, voltages[:, i], label='target')
      ax2.plot(times, vs[:, i], label=label)
      plt.legend(fontsize=10)
      if i == 0:
        plt.ylabel('Potential [mV]')
      if xlabel:
        plt.xlabel('Time [ms]')
    return ax_idx + 3

  param_bfgs, _ = fitting_by_gradient(fit_target, n_sample=100)

  currents = inp_traces.T
  times = np.arange(currents.shape[0]) * bst.environ.get_dt()
  currents = np.asarray(currents)
  voltages = np.asarray(target_hists[1])

  others = ['DE', 'PSO', 'TwoPointsDE', 'bayesian']
  fig, gs = bts.visualize.get_figure(5 + 3 * len(others), currents.shape[1], 0.8, 4.5)
  ax_idx = 0
  ax_idx = add_current(fig, gs, ax_idx)
  ax_idx = add_plot(param_bfgs, fig, gs, ax_idx, 'L-BFGS-B')
  for i, method in enumerate(others):
    param, _ = fitting_by_others(fit_target, method=method)
    ax_idx = add_plot(param, fig, gs, ax_idx, label=method, xlabel=i == len(others) - 1)
  plt.show()


if __name__ == '__main__':
  pass
  # visualize_fitting_trarget()
  # fitting_by_gradient()
  # fitting_by_others()
  compare_different_fitting_v2('spike')
  # compare_different_fitting_v2('potential')
