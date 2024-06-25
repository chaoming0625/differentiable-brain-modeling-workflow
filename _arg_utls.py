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


import argparse
import os
from typing import Sequence, Union

os.environ['JAX_TRACEBACK_FILTERING'] = 'off'


__all__ = [
  'MyArgumentParser'
]


def _disable_gpu_memory_preallocation(release_memory: bool = True):
  """Disable pre-allocating the GPU memory.

  This disables the preallocation behavior. JAX will instead allocate GPU memory as needed,
  potentially decreasing the overall memory usage. However, this behavior is more prone to
  GPU memory fragmentation, meaning a JAX program that uses most of the available GPU memory
  may OOM with preallocation disabled.

  Args:
    release_memory: bool. Whether we release memory during the computation.
  """
  os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
  if release_memory:
    os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'


def _enable_gpu_memory_preallocation():
  """Disable pre-allocating the GPU memory."""
  os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'
  os.environ.pop('XLA_PYTHON_CLIENT_ALLOCATOR', None)


def set_gpu_preallocation(mode: Union[float, bool]):
  """GPU memory allocation.

  If preallocation is enabled, this makes JAX preallocate ``percent`` of the total GPU memory,
  instead of the default 75%. Lowering the amount preallocated can fix OOMs that occur when the JAX program starts.
  """
  if mode is False:
    _disable_gpu_memory_preallocation()
    return
  if mode is True:
    _enable_gpu_memory_preallocation()
    return
  assert isinstance(mode, float) and 0. <= mode < 1., f'GPU memory preallocation must be in [0., 1.]. But got {mode}.'
  os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(mode)


def set_gpu_device(device_ids: Union[str, int, Sequence[int]]):
  if isinstance(device_ids, int):
    device_ids = str(device_ids)
  elif isinstance(device_ids, (tuple, list)):
    device_ids = ','.join([str(d) for d in device_ids])
  elif isinstance(device_ids, str):
    device_ids = device_ids
  else:
    raise ValueError
  os.environ['CUDA_VISIBLE_DEVICES'] = device_ids


def _set_device(parser) -> argparse.ArgumentParser:
  return parser


def _add_training_method(parser) -> argparse.ArgumentParser:
  return parser


class MyArgumentParser(argparse.ArgumentParser):
  def __init__(self, *args, **kwargs):
    super(MyArgumentParser, self).__init__(*args, **kwargs)
    self.add_argument('--devices', type=str, default='0', help='The GPU device ids.')
    self.add_argument("--memory_preallocate", type=int, default=1, help="The ratio for network simulation.")
    self.add_argument("--method", type=str, default='bptt', help="Training method.")

    # device setting
    args, _ = self.parse_known_args()
    set_gpu_device(args.devices)
    set_gpu_preallocation(0.99)
    if args.memory_preallocate == 1:
      _disable_gpu_memory_preallocation()

    # training algorithms
    if args.method != 'bptt':
      self.add_argument("--diag_normalize", type=int, default=0, choices=[0, 1, 2],
                        help="Normalize the diagonal jacobian (0 - None, 1 - True, 2 - False).")
      self.add_argument("--diag_jacobian", type=str, default='exact', help="Normalize the diagonal jacobian.")
      if args.method != 'diag':
        self.add_argument("--etrace_decay", type=float, default=0.9, help="The time constant of eligibility trace ")
        self.add_argument("--num_snap", type=int, default=0, help="The number of snap shoot.")
        self.add_argument("--snap_freq", type=int, default=None, help="The frequency of snap shoot.")

