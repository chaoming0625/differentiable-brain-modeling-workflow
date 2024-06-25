import abc
from typing import Callable, Optional, Dict, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np
from nevergrad import p
from nevergrad.optimization import optimizerlib, registry
from scipy.optimize import minimize
from sklearn.base import RegressorMixin
from skopt import Optimizer as skoptOptimizer
from skopt.space import Real
from tqdm.auto import tqdm

__all__ = [
  'Optimizer',
  'BFGSOptimizer',
  'NevergradOptimizer',
  'ScipyOptimizer',
  'SkoptOptimizer',
  'gamma_factor',
]


class Optimizer(metaclass=abc.ABCMeta):
  """
  Optimizer class created as a base for optimization initialization and
  performance with different libraries. To be used with modelfitting
  Fitter.
  """

  @abc.abstractmethod
  def initialize(self, *args, **kwargs):
    """
    Initialize the instrumentation for the optimization, based on
    parameters, creates bounds for variables and attaches them to the
    optimizer
    """
    pass

  @abc.abstractmethod
  def one_trial(self, *args, **kwargs):
    """
    Returns the requested number of samples of parameter sets

    Parameters
    ----------
    n_samples: int
        number of samples to be drawn

    Returns
    -------
    parameters: list
        list of drawn parameters [n_samples x n_params]
    """
    pass

  def minimize(self, n_iter):
    results = []
    bar = tqdm(total=n_iter)
    for i in range(n_iter):
      r = self.one_trial(choice_best=True)
      results.append(r)
      bar.update()
      bar.set_description(f'Current best error: {np.nanmin(self.errors):.5f}')
    return results[-1]


class NevergradOptimizer(Optimizer):
  """
  ``NevergradOptimizer`` instance creates all the tools necessary for the user
  to use it with Nevergrad library.

  Parameters
  ----------
  method: `str`, optional
      The optimization method. By default differential evolution, can be
      chosen from any method in Nevergrad registry
  use_nevergrad_recommendation: bool, optional
      Whether to use Nevergrad's recommendation as the "best result". This
      recommendation takes several evaluations of the same parameters (for
      stochastic simulations) into account. The alternative is to simply
      return the parameters with the lowest error so far (the default). The
      problem with Nevergrad's recommendation is that it can give wrong result
      for errors that are very close in magnitude due (see github issue #16).
  budget: int or None
      number of allowed evaluations
  num_workers: int
      number of evaluations which will be run in parallel at once
  """

  def __init__(
      self,
      loss_fun: Callable,
      n_sample: int,
      bounds: Optional[Union[Sequence, Dict]] = None,
      method: str = 'DE',
      use_nevergrad_recommendation: bool = False,
      **kwargs
  ):
    super(Optimizer, self).__init__()

    # loss function to evaluate
    assert callable(loss_fun), "'loss_fun' must be a callable function"
    self.loss_fun = loss_fun

    # population size
    assert n_sample > 0, "'n_sample' must be a positive integer"
    self.n_sample = n_sample

    # optimization method
    if method not in registry:
      raise AssertionError(f"Unknown to Nevergrad optimization method: {method}")
    self.method = method

    # bounds
    if bounds is None:
      bounds = ()
    self.bounds = bounds

    # others
    self.use_nevergrad_recommendation = use_nevergrad_recommendation
    self.kwds = kwargs

  def initialize(self):
    self.tested_parameters = []
    self.errors = []
    if isinstance(self.bounds, dict):
      parameters = dict()
      for key, bound in self.bounds.items():
        assert len(bound) == 2, f'Each bound must be a tuple of two elements, got {bound}'
        n_size = np.size(bound[0])
        p_ = p.Scalar(lower=float(bound[0]), upper=float(bound[1]))
        parameters[key] = p_
      parametrization = p.Dict(**parameters)
    elif isinstance(self.bounds, (list, tuple)):  # 目前都转换为字典                   了
      parameters = dict()
      for i, bound in enumerate(self.bounds):
        assert len(bound) == 2, f'Each bound must be a tuple of two elements, got {bound}'
        p_ = p.Scalar(lower=float(bound[0]), upper=float(bound[1]))
        parameters[f'key_{i}'] = p_
      parametrization = p.Dict(**parameters)
    else:
      raise ValueError(f"Unknown type of 'bounds': {type(self.bounds)}")
    self.optim = optimizerlib.registry[self.method](parametrization=parametrization, **self.kwds)
    self.optim._llambda = self.n_sample

  def one_trial(self, choice_best: bool = False):
    # draw parameters
    candidates = [self.optim.ask() for _ in range(self.n_sample)]
    parameters = [list(cand.value.values()) for cand in candidates]
    parameters2 = np.asarray(parameters).T  # (num_param, num_n_sample)

    # evaluate the parameters
    errors = self.loss_fun(*parameters2)
    errors = np.asarray(errors)

    # tell the optimizer
    assert len(parameters) == len(errors), "Number of parameters and errors must be the same"
    for candidate, error in zip(candidates, errors):
      self.optim.tell(candidate, error)

    # record the tested parameters and errors
    self.tested_parameters.extend(parameters)
    self.errors.extend(list(errors))

    # return the best parameter
    if choice_best:
      if self.use_nevergrad_recommendation:
        res = self.optim.provide_recommendation()
        return res.args
      else:
        best = np.nanargmin(self.errors)
        return self.tested_parameters[best]


class SkoptOptimizer(Optimizer):
  """
  SkoptOptimizer instance creates all the tools necessary for the user
  to use it with scikit-optimize library.

  Parameters
  ----------
  parameter_names: list[str]
      Parameters to be used as instruments.
  bounds : list
      List with appropiate bounds for each parameter.
  method : `str`, optional
      The optimization method. Possibilities: "GP", "RF", "ET", "GBRT" or
      sklearn regressor, default="GP"
  n_calls: `int`
      Number of calls to ``func``. Defaults to 100.
  """

  def __init__(
      self,
      loss_fun: Callable,
      n_sample: int,
      bounds: Optional[Sequence] = None,
      method='GP',
      **kwds
  ):
    super(Optimizer, self).__init__()

    # loss function
    assert callable(loss_fun), "'loss_fun' must be a callable function"
    self.loss_fun = loss_fun

    # method
    if not (method.upper() in ["GP", "RF", "ET", "GBRT"] or isinstance(method, RegressorMixin)):
      raise AssertionError(f"Provided method: {method} is not an skopt optimization or a regressor")
    self.method = method

    # population size
    assert n_sample > 0, "'n_sample' must be a positive integer"
    self.n_sample = n_sample

    # bounds
    if bounds is None:
      bounds = ()
    self.bounds = bounds

    # others
    self.kwds = kwds

  def initialize(self):
    self.tested_parameters = []
    self.errors = []
    instruments = []
    for bound in self.bounds:
      instrumentation = Real(*np.asarray(bound), transform='normalize')
      instruments.append(instrumentation)
    self.optim = skoptOptimizer(dimensions=instruments, base_estimator=self.method, **self.kwds)

  def one_trial(self, choice_best: bool = False):
    # draw parameters
    parameters = self.optim.ask(n_points=self.n_sample)
    self.tested_parameters.extend(parameters)

    # errors
    errors = self.loss_fun(*np.asarray(parameters).T)
    errors = np.asarray(errors).tolist()
    self.errors.extend(errors)

    # tell
    # for parameter, error in zip(parameters, errors):
    #   error = float(error)
    #   # Non-sense values including NaNs should not be accepted.
    #   # We do not use max-float as various later transformations could lead to greater values.
    #   if not error < 5.0e20:  # pylint: disable=unneeded-not
    #       # self._warn(
    #       #     f"Clipping very high value {error} in tell (rescale the cost function?).",
    #       #     errors.LossTooLargeWarning,
    #       # )
    #       error = 5.0e20  # sys.float_info.max leads to numerical problems so let us do this.
    #   self.optim.tell(parameter, error)

    errors = np.array(errors)
    errors[errors > 5.0e20] = 5.0e20
    errors = errors.tolist()
    errors = np.nan_to_num(errors, nan=5.0e20).tolist()
    self.optim.tell(parameters, errors)

    if choice_best:
      xi = self.optim.Xi
      yii = np.array(self.optim.yi)
      return xi[yii.argmin()]


def gamma_factor(model, data, delta=0.01, rate_correction=True, dt=0.1):
  # model: [n_time]
  # data: [n_time]

  # JIT:
  # 1. shape known, consistent
  # 2. no python control flow, where, jax.lax.cond

  time = model.shape[0] * dt / 1000  # total time of the simulation, [s]
  model_spk_time = jnp.where(model, size=model.shape[0], fill_value=np.inf)[0] * dt  # model spiking time
  data_spk_time = jnp.where(data, size=model.shape[0], fill_value=np.inf)[0] * dt  # given data spiking time
  delta_length = jnp.rint(delta / dt)
  model_length = jnp.asarray(jnp.sum(model), dtype=jnp.int32)
  data_length = jnp.asarray(jnp.sum(data), dtype=jnp.int32)

  bins = .5 * (model_spk_time[1:] + model_spk_time[:-1])
  indices = jnp.digitize(data_spk_time, bins)
  diff = jnp.abs(data_spk_time - model_spk_time[indices])
  matched_spikes = (diff <= delta_length)
  coincidences = jnp.sum(matched_spikes)

  data_rate = data_length / time  # firing rate of the data, Hz
  model_rate = model_length / time  # firing rate of the model, Hz

  # Normalization of the coincidences count
  normalized_coin = 2 * delta * data_length * data_rate
  norm = .5 * (1 - 2 * data_rate * delta)
  # 为防止除0错误胡写：这个if else是自己规定的，最后一个else是原本的
  gamma = (coincidences - normalized_coin) / (norm * (model_length + data_length))
  gamma = jnp.where(jnp.logical_and(data_length == 0, model_length == 0),
                    1,
                    jnp.where(jnp.logical_and(data_length == 0, model_length != 0),
                              0,
                              gamma))
  # 为防止除0错误胡写,从elif后都是对的
  rate_term = jnp.where(data_rate == 0,
                        1,
                        1 + 2 * jnp.abs((data_rate - model_rate) / data_rate) if rate_correction else 1)
  # return rate_term - gamma
  return jnp.clip(rate_term - gamma, 0., np.inf)


class ScipyOptimizer(Optimizer):
  """
  A simple wrapper for scipy.optimize.minimize using JAX.

  Parameters
  ----------
  fun: function
    The objective function to be minimized, written in JAX code
    so that it is automatically differentiable.  It is of type,
    ```fun: x, *args -> float``` where `x` is a PyTree and args
    is a tuple of the fixed parameters needed to completely specify the function.

  x0: jnp.ndarray
    Initial guess represented as a JAX PyTree.

  args: tuple, optional.
    Extra arguments passed to the objective function
    and its derivative.  Must consist of valid JAX types; e.g. the leaves
    of the PyTree must be floats.

  method : str or callable, optional
    Type of solver.  Should be one of
        - 'Nelder-Mead' :ref:`(see here) <optimize.minimize-neldermead>`
        - 'Powell'      :ref:`(see here) <optimize.minimize-powell>`
        - 'CG'          :ref:`(see here) <optimize.minimize-cg>`
        - 'BFGS'        :ref:`(see here) <optimize.minimize-bfgs>`
        - 'Newton-CG'   :ref:`(see here) <optimize.minimize-newtoncg>`
        - 'L-BFGS-B'    :ref:`(see here) <optimize.minimize-lbfgsb>`
        - 'TNC'         :ref:`(see here) <optimize.minimize-tnc>`
        - 'COBYLA'      :ref:`(see here) <optimize.minimize-cobyla>`
        - 'SLSQP'       :ref:`(see here) <optimize.minimize-slsqp>`
        - 'trust-constr':ref:`(see here) <optimize.minimize-trustconstr>`
        - 'dogleg'      :ref:`(see here) <optimize.minimize-dogleg>`
        - 'trust-ncg'   :ref:`(see here) <optimize.minimize-trustncg>`
        - 'trust-exact' :ref:`(see here) <optimize.minimize-trustexact>`
        - 'trust-krylov' :ref:`(see here) <optimize.minimize-trustkrylov>`
        - custom - a callable object (added in version 0.14.0),
          see below for description.
    If not given, chosen to be one of ``BFGS``, ``L-BFGS-B``, ``SLSQP``,
    depending on if the problem has constraints or bounds.

  bounds : sequence or `Bounds`, optional
    Bounds on variables for L-BFGS-B, TNC, SLSQP, Powell, and
    trust-constr methods. There are two ways to specify the bounds:
        1. Instance of `Bounds` class.
        2. Sequence of ``(min, max)`` pairs for each element in `x`. None
        is used to specify no bound.
    Note that in order to use `bounds` you will need to manually flatten
    them in the same order as your inputs `x0`.

  constraints : {Constraint, dict} or List of {Constraint, dict}, optional
    Constraints definition (only for COBYLA, SLSQP and trust-constr).
    Constraints for 'trust-constr' are defined as a single object or a
    list of objects specifying constraints to the optimization problem.
    Available constraints are:
        - `LinearConstraint`
        - `NonlinearConstraint`
    Constraints for COBYLA, SLSQP are defined as a list of dictionaries.
    Each dictionary with fields:
        type : str
            Constraint type: 'eq' for equality, 'ineq' for inequality.
        fun : callable
            The function defining the constraint.
        jac : callable, optional
            The Jacobian of `fun` (only for SLSQP).
        args : sequence, optional
            Extra arguments to be passed to the function and Jacobian.
    Equality constraint means that the constraint function result is to
    be zero whereas inequality means that it is to be non-negative.
    Note that COBYLA only supports inequality constraints.

    Note that in order to use `constraints` you will need to manually flatten
    them in the same order as your inputs `x0`.

  tol : float, optional
    Tolerance for termination. For detailed control, use solver-specific
    options.

  options : dict, optional
      A dictionary of solver options. All methods accept the following
      generic options:
          maxiter : int
              Maximum number of iterations to perform. Depending on the
              method each iteration may use several function evaluations.
          disp : bool
              Set to True to print convergence messages.
      For method-specific options, see :func:`show_options()`.

  callback : callable, optional
      Called after each iteration. For 'trust-constr' it is a callable with
      the signature:
          ``callback(xk, OptimizeResult state) -> bool``
      where ``xk`` is the current parameter vector represented as a PyTree,
       and ``state`` is an `OptimizeResult` object, with the same fields
      as the ones from the return. If callback returns True the algorithm
      execution is terminated.

      For all the other methods, the signature is:
          ```callback(xk)```
      where `xk` is the current parameter vector, represented as a PyTree.

  Returns
  -------
  res : The optimization result represented as a ``OptimizeResult`` object.
    Important attributes are:
        ``x``: the solution array, represented as a JAX PyTree
        ``success``: a Boolean flag indicating if the optimizer exited successfully
        ``message``: describes the cause of the termination.
    See `scipy.optimize.OptimizeResult` for a description of other attributes.

  """

  def __init__(
      self,
      loss_fun: Callable,
      bounds: np.ndarray | Sequence,
      method: str = 'L-BFGS-B',
      constraints=(),
      tol=None,
      callback=None,
      options=None,
  ):
    self.loss_fun = jax.jit(loss_fun)
    self.method = method
    self.bounds = bounds
    assert len(bounds) == 2, "Bounds must be a tuple of two elements: (min, max)"
    self.constraints = constraints
    self.tol = tol
    self.callback = callback
    self.options = options

    # Wrap the gradient in a similar manner
    self.jac = jax.jit(jax.grad(loss_fun))

  def one_trial(self, *args, **kwargs):
    pass

  def initialize(self, *args, **kwargs):
    pass

  def minimize(self, num_sample=1):
    bounds = np.asarray(self.bounds).T
    xs = np.random.uniform(self.bounds[0], self.bounds[1], size=(num_sample,) + self.bounds[0].shape)
    best_l = np.inf
    best_r = None

    for x0 in xs:
      results = minimize(self.loss_fun,
                         x0,
                         method=self.method,
                         jac=self.jac,
                         callback=self.callback,
                         bounds=bounds,
                         constraints=self.constraints,
                         tol=self.tol,
                         options=self.options)
      if results.fun < best_l:
        best_l = results.fun
        best_r = results
    return best_r


from jax.scipy.optimize import minimize as jax_minimize


class BFGSOptimizer(Optimizer):
  def __init__(
      self,
      loss_fun: Callable,
      bounds: np.ndarray | Sequence,
      constraints=(),
      tol=None,
      callback=None,
      options=None,
      bound_factor: float = 1.0,
  ):
    self.loss_fun = loss_fun
    self.bounds = bounds
    assert len(bounds) == 2, "Bounds must be a tuple of two elements: (min, max)"
    self.constraints = constraints
    self.tol = tol
    self.callback = callback
    self.options = options
    self.bound_factor = bound_factor

    # Wrap the gradient in a similar manner
    self.jac = jax.jit(jax.grad(self._true_loss))
    self.run_fun = jax.jit(jax.vmap(self._minimize))

  def _true_loss(self, x0):
    l = self.loss_fun(x0)
    l += jnp.sum(jax.nn.relu(self.bounds[0] - x0) ** 2) * self.bound_factor
    l += jnp.sum(jax.nn.relu(x0 - self.bounds[1]) ** 2) * self.bound_factor
    return l

  # @functools.partial(jax.jit, static_argnums=(0,))
  def _minimize(self, x0):
    return jax_minimize(self._true_loss, x0, method='bfgs', tol=self.tol, options=self.options)

  def one_trial(self, *args, **kwargs):
    pass

  def initialize(self, *args, **kwargs):
    pass

  def minimize(self, num_sample, select='loss'):
    bounds = np.asarray(self.bounds)
    batch_x0 = np.random.uniform(bounds[0], bounds[1], size=(num_sample, bounds.shape[1]))
    if select == 'loss':
      res = self.run_fun(batch_x0)
      idx = jnp.argmin(res.fun)
      return jax.tree.map(lambda x: x[idx] if x is not None else x, res)
    else:
      raise ValueError(f"Unknown selection method: {select}")
