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


import time
from functools import partial
from typing import Union, Callable, Optional

import brainstate as bst
import braintools as bts
import brainunit as bu
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _utils import gamma_factor, NevergradOptimizer, ScipyOptimizer, SkoptOptimizer

bst.environ.set(dt=0.01)

# Load Input and Output Data
df_inp_traces = pd.read_csv('neuron_data/input_traces_hh.csv')
df_out_traces = pd.read_csv('neuron_data/output_traces_hh.csv')

inp_traces = df_inp_traces.to_numpy()
inp_traces = inp_traces[:, 1:] * 1e9


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


class HHLTC(bst.nn.Neuron):
  def __init__(
      self,
      size: bst.typing.Size,
      keep_size: bool = False,
      mode: bst.mixin.Mode = None,
      name: str = None,

      # neuron parameters
      ENa: Union[bst.typing.ArrayLike, Callable] = 50.,
      gNa: Union[bst.typing.ArrayLike, Callable] = 120.,
      EK: Union[bst.typing.ArrayLike, Callable] = -77.,
      gK: Union[bst.typing.ArrayLike, Callable] = 36.,
      EL: Union[bst.typing.ArrayLike, Callable] = -54.387,
      gL: Union[bst.typing.ArrayLike, Callable] = 0.03,
      V_th: Union[bst.typing.ArrayLike, Callable] = 20.,
      C: Union[bst.typing.ArrayLike, Callable] = 1.0,
      V_initializer: Callable = bst.init.Uniform(-70, -60.),
      m_initializer: Optional[Union[Callable, bst.typing.ArrayLike]] = None,
      h_initializer: Optional[Union[Callable, bst.typing.ArrayLike]] = None,
      n_initializer: Optional[Union[Callable, bst.typing.ArrayLike]] = None,
      spk_fun: Callable = bst.surrogate.ReluGrad(),
  ):
    # initialization
    super().__init__(size, keep_size=keep_size, mode=mode, name=name, spk_fun=spk_fun)

    # parameters
    self.ENa = bst.init.param(ENa, self.varshape)
    self.EK = bst.init.param(EK, self.varshape)
    self.EL = bst.init.param(EL, self.varshape)
    self.gNa = bst.init.param(gNa, self.varshape)
    self.gK = bst.init.param(gK, self.varshape)
    self.gL = bst.init.param(gL, self.varshape)
    self.C = bst.init.param(C, self.varshape)
    self.V_th = bst.init.param(V_th, self.varshape)

    # initializers
    self._m_initializer = m_initializer
    self._h_initializer = h_initializer
    self._n_initializer = n_initializer
    self._V_initializer = V_initializer

  # m channel
  m_alpha = lambda self, V: 0.32 * 4 / bu.math.exprel((13. - V + self.V_th) / 4.)
  m_beta = lambda self, V: 0.28 * 5 / bu.math.exprel((V - self.V_th - 40.) / 5.)
  m_inf = lambda self, V: self.m_alpha(V) / (self.m_alpha(V) + self.m_beta(V))
  dm = lambda self, m, t, V: self.m_alpha(V) * (1 - m) - self.m_beta(V) * m

  # h channel
  h_alpha = lambda self, V: 0.128 * bu.math.exprel((17. - V + self.V_th) / 18.)
  h_beta = lambda self, V: 4. / (1 + jnp.exp((40. - V + self.V_th) / 5.))
  h_inf = lambda self, V: self.h_alpha(V) / (self.h_alpha(V) + self.h_beta(V))
  dh = lambda self, h, t, V: self.h_alpha(V) * (1 - h) - self.h_beta(V) * h

  # n channel
  n_alpha = lambda self, V: 0.032 * 5 / bu.math.exprel((15. - V + self.V_th) / 5.)
  n_beta = lambda self, V: .5 * jnp.exp((10. - V + self.V_th) / 40.)
  n_inf = lambda self, V: self.n_alpha(V) / (self.n_alpha(V) + self.n_beta(V))
  dn = lambda self, n, t, V: self.n_alpha(V) * (1 - n) - self.n_beta(V) * n

  def dV(self, V, t, m, h, n, I):
    I = self.sum_current_inputs(V, init=I)
    I_Na = (self.gNa * m * m * m * h) * (V - self.ENa)
    n2 = n * n
    I_K = (self.gK * n2 * n2) * (V - self.EK)
    I_leak = self.gL * (V - self.EL)
    dVdt = (- I_Na - I_K - I_leak + I) / self.C
    return dVdt

  def init_state(self, batch_size=None):
    self.V = bst.ShortTermState(bst.init.param(self._V_initializer, self.varshape, batch_size))
    self.m = bst.ShortTermState(bst.init.param(self._m_initializer, self.varshape, batch_size))
    self.h = bst.ShortTermState(bst.init.param(self._h_initializer, self.varshape, batch_size))
    self.n = bst.ShortTermState(bst.init.param(self._n_initializer, self.varshape, batch_size))

  def update(self, x=0.):
    t = bst.environ.get('t')
    last_V = self.V.value
    V = bst.nn.exp_euler_step(self.dV, last_V, t, self.m.value, self.h.value, self.n.value, x)
    m = bst.nn.exp_euler_step(self.dm, self.m.value, t, last_V)
    h = bst.nn.exp_euler_step(self.dh, self.h.value, t, last_V)
    n = bst.nn.exp_euler_step(self.dn, self.n.value, t, last_V)
    V += self.sum_delta_inputs()
    self.V.value = V
    self.m.value = m
    self.h.value = h
    self.n.value = n
    return self.get_spike(last_V, V)

  def get_spike(self, last_V, V):
    return self.spk_fun(last_V - self.V_th) * self.spk_fun(V - self.V_th)


@bst.transform.jit
def simulate_model(current, gl, g_na, g_kd, vth):
  assert current.ndim == 2  # [T, B]
  n_input = current.shape[1]
  hh = HHLTC(
    n_input, gL=gl, gNa=g_na, gK=g_kd, C=0.2, EL=-65, EK=-90, ENa=50, V_th=vth,
    V_initializer=bst.init.Constant(-65.),
    m_initializer=bst.init.Constant(0.),
    h_initializer=bst.init.Constant(0.),
    n_initializer=bst.init.Constant(0.),
  )
  hh.init_state()

  def step_fun(i, inp):
    with bst.environ.context(i=i, t=bst.environ.get_dt() * i):
      spk = hh.update(inp)
    return spk, hh.V.value, hh.m.value, hh.n.value, hh.h.value

  indices = np.arange(current.shape[0])
  return bst.transform.for_loop(step_fun, indices, current)  # (T, B)


def compare_spikes(param, currents, target_spks):
  gl, g_na, g_kd, vth = param
  spks = simulate_model(currents, gl, g_na, g_kd, vth)[0]  # (T, B)
  losses = jax.vmap(partial(gamma_factor, dt=bst.environ.get_dt()), axis_size=1)(spks, target_spks)
  return losses.sum()


def compare_potentials(param, currents, target_potentials, n_point=10):
  gl, g_na, g_kd, vth = param
  vs = simulate_model(currents, gl, g_na, g_kd, vth)[1]  # (T, B)
  # indices = np.random.randint(0, vs.shape[0], vs.shape[0] // n_point)
  indices = np.arange(0, vs.shape[0], vs.shape[0] // n_point)
  losses = bts.metric.squared_error(vs[indices], target_potentials[indices])
  return losses.mean()


# inp_traces: [B, T]
target_hists = simulate_model(inp_traces.T,
                              0.008363657201143428,
                              24.697737655778038,
                              5.814895041093182,
                              -55.31094455202257)


def visualize_hh_input_and_output():
  # Load Input and Output Data
  inp_traces = df_inp_traces.to_numpy()
  inp_traces = inp_traces[:, 1:] * 1e9

  out_traces = df_out_traces.to_numpy()
  out_traces = out_traces[:, 1:]

  indices = np.arange(inp_traces.shape[1]) * 0.01

  fig, gs = bts.visualize.get_figure(3, 1, 1.2, 6.0)
  ax = fig.add_subplot(gs[0, 0])
  ax.plot(indices, inp_traces.T)
  plt.xticks([])
  plt.ylabel('Current [nA]', fontsize=13)

  ax2 = fig.add_subplot(gs[1:, 0])
  ax2.plot(indices, out_traces.T)
  plt.ylabel('Potential [mV]', fontsize=13)
  plt.xlabel('Time [ms]')

  fig.align_ylabels([ax, ax2])
  plt.show()


bounds = [np.asarray([0.0001, 0.01, 0.01, -65.]),
          np.asarray([0.1, 40., 20., -50.])]


def fitting_by_gradient(fit_target='spike', n_sample=20):
  print(f"Method: L-BFGS-B, fit_target: {fit_target}, n_sample: {n_sample}")

  if fit_target == 'spike':
    fun = jax.jit(partial(compare_spikes, currents=inp_traces.T, target_spks=target_hists[0]))
  elif fit_target == 'potential':
    fun = jax.jit(lambda x: compare_potentials(x, inp_traces.T, target_hists[1], n_point=10))
  else:
    raise ValueError(f"Unknown fit target: {fit_target}")

  # opt = BFGSOptimizer(fun, bounds=bounds, bound_factor=1.)
  opt = ScipyOptimizer(fun, bounds=bounds)
  param = opt.minimize(num_sample=n_sample)

  print(param.x)
  print(param.fun)
  # visualize(inp_traces.T, target_hists[1], *param.x)
  return param.x, param.fun


def fitting_by_others(fit_target='spike', method='DE', n_sample=20):
  print(f"Method: {method}, fit_target: {fit_target}, n_sample: {n_sample}")

  if fit_target == 'spike':
    fun = jax.jit(partial(compare_spikes, currents=inp_traces.T, target_spks=target_hists[0]))
  elif fit_target == 'potential':
    fun = jax.jit(partial(compare_potentials, currents=inp_traces.T, target_potentials=target_hists[1]))
  else:
    raise ValueError(f"Unknown fit target: {fit_target}")

  @jax.jit
  @jax.vmap
  def loss_with_multiple_run(gl, g_na, g_kd, vth):
    return fun((gl, g_na, g_kd, vth))

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
      bounds={'gl': (0.0001, 0.1), 'g_na': (0.01, 40), 'g_kd': (0.01, 20), 'vth': (-65, -50)},
      use_nevergrad_recommendation=False,
      method=method,
      # budget=100,
    )
  opt.initialize()
  param = opt.minimize(5)
  loss = fun(param)
  print(param)
  print(loss)
  # visualize(inp_traces.T, target_hists[1], *param)
  return param, loss


def compute_times(fit_target='spike', n=10):
  losses = dict(gradient=[])
  times = dict(gradient=[])
  others = ['DE', 'PSO', 'TwoPointsDE', 'bayesian']
  for method in others:
    losses[method] = []
    times[method] = []

  for _ in range(n):
    t0 = time.time()
    param1, l1 = fitting_by_gradient(fit_target, n_sample=20)
    print(f"Gradient fitting time: {time.time() - t0:.3f}s")
    losses['gradient'].append(l1)
    times['gradient'].append(time.time() - t0)

    for method in others:
      t0 = time.time()
      param2, l2 = fitting_by_others(fit_target, method=method, n_sample=20)
      print(f"{method} fitting time: {time.time() - t0:.3f}s")
      losses[method].append(l2)
      times[method].append(time.time() - t0)

  for k in list(losses.keys()):
    losses[k] = np.asarray(losses[k])
    times[k] = np.asarray(times[k])
  print(jax.tree.map(np.mean, losses), jax.tree.map(np.std, losses))
  print(jax.tree.map(np.mean, times), jax.tree.map(np.std, times))
  print()


def compare_different_fitting(fit_target='spike'):
  # plt.style.use(['science', 'nature', ])

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

  param_bfgs, _ = fitting_by_gradient(fit_target)

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
    if i == len(others) - 1:
      ax_idx = add_plot(param, fig, gs, ax_idx, label=method, xlabel=True)
    else:
      ax_idx = add_plot(param, fig, gs, ax_idx, label=method)
  plt.show()


if __name__ == '__main__':
  pass
  # visualize(inp_traces.T, target_hists[1],
  #           0.008363657201143428, 24.697737655778038,
  #           5.814895041093182, -55.31094455202257)
  # visualize_hh_input_and_output()

  # fitting_by_gradient()
  # fitting_by_others()
  # compare_different_fitting('spike')
  # compute_times('potential')
  # compare_different_fitting('potential')
