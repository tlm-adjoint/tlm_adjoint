#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# For tlm_adjoint copyright information see ACKNOWLEDGEMENTS in the tlm_adjoint
# root directory

# This file is part of tlm_adjoint.
# 
# tlm_adjoint is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, version 3 of the License.
# 
# tlm_adjoint is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public License
# along with tlm_adjoint.  If not, see <https://www.gnu.org/licenses/>.

# This file implements binomial checkpointing using the approach described in
#   GW2000  A. Griewank and A. Walther, "Algorithm 799: Revolve: An
#           implementation of checkpointing for the reverse or adjoint mode of
#           computational differentiation", ACM Transactions on Mathematical
#           Software, 26(1), pp. 19--45, 2000
# choosing the maximum step size permitted when choosing the next snapshot
# block. This file further implements multi-stage offline checkpointing,
# determined via a brute force search to yield behaviour described in
#   SW2009  P. Stumm and A. Walther, "MultiStage approaches for optimal offline
#           checkpointing", SIAM Journal on Scientific Computing, 31(3),
#           pp. 1946--1967, 2009

from collections import defaultdict
import numpy

__all__ = \
  [
    "CheckpointingException",
    "MultistageManager"
  ]
  
class CheckpointingException(Exception):
  pass
    
def n_advance(n, snapshots):
  """
  Determine an optimal offline snapshot interval, taking n steps and with the
  given number of snapshots, using the approach of
    GW2000  A. Griewank and A. Walther, "Algorithm 799: Revolve: An
            implementation of checkpointing for the reverse or adjoint mode of
            computational differentiation", ACM Transactions on Mathematical
            Software, 26(1), pp. 19--45, 2000
  and choosing the maximal possible step size.
  """

  if n < 1:
    raise CheckpointingException("Require at least one block")
  if snapshots <= 0:
    raise CheckpointingException("Require at least one snapshot")
  
  # Discard excess snapshots
  snapshots = max(min(snapshots, n - 1), 1)  
  # Handle limiting cases
  if snapshots == 1:
    return n - 1  # Minimal storage
  elif snapshots == n - 1:
    return 1  # Maximal storage
  
  # Find t as in GW2000 Proposition 1 (note 'm' in GW2000 is 'n' here, and 's'
  # in GW2000 is 'snapshots' here). Compute values of beta as in equation (1) of
  # GW2000 as a side effect. We must have a minimal rerun of at least 2 (the
  # minimal rerun of 1 case is maximal storage, handled above) so we start from
  # t = 2.
  t = 2
  b_s_tm2 = 1
  b_s_tm1 = snapshots + 1
  b_s_t = ((snapshots + 1) * (snapshots + 2)) // 2
  while b_s_tm1 >= n or n > b_s_t:
    t += 1
    b_s_tm2, b_s_tm1, b_s_t = b_s_tm1, b_s_t, (b_s_t * (snapshots + t)) // t
  
  # Return the maximal step size compatible with Fig. 4 of GW2000
  b_sm1_tm2 = (b_s_tm2 * snapshots) // (snapshots + t - 2)
  if n <= b_s_tm1 + b_sm1_tm2:
    return n - b_s_tm1 + b_s_tm2
  b_sm1_tm1 = (b_s_tm1 * snapshots) // (snapshots + t - 1)
  b_sm2_tm1 = (b_sm1_tm1 * (snapshots - 1)) // (snapshots + t - 2)
  if n <= b_s_tm1 + b_sm2_tm1:
    return b_s_tm2 + b_sm1_tm2
  elif n <= b_s_tm1 + b_sm1_tm1 + b_sm2_tm1:
    return n - b_sm1_tm1 - b_sm2_tm1
  else:
    return  b_s_tm1

def allocate_snapshots(max_n, snapshots_in_ram, snapshots_on_disk, write_weight = 1.0, read_weight = 1.0, delete_weight = 0.0):
  """
  Allocate a stack of snapshots based upon the number of read/writes,
  preferentially allocating to RAM. Yields the approach described in
    SW2009  P. Stumm and A. Walther, "MultiStage approaches for optimal offline
            checkpointing", SIAM Journal on Scientific Computing, 31(3),
            pp. 1946--1967, 2009  
  but applies a brute force approach to determine the allocation.
  """

  snapshots = snapshots_in_ram + snapshots_on_disk
  snapshots_n = []
  weights = numpy.zeros(snapshots, dtype = numpy.float64)
  n = 0
  i = 0
  snapshots_n.append(n)
  weights[i] += write_weight
  while True:
    n += n_advance(max_n - n, snapshots - i)
    if n == max_n - 1:
      break
    i += 1
    snapshots_n.append(n)
    weights[i] += write_weight
  while n > 0:
    snapshot_n_0 = snapshot_n = snapshots_n[-1]
    weights[i] += read_weight
    while True:
      snapshot_n += n_advance(n - snapshot_n, snapshots - i)
      if snapshot_n == n - 1:
        break
      i += 1
      snapshots_n.append(snapshot_n)
      weights[i] += write_weight
    if snapshot_n_0 == n - 1:
      snapshots_n.pop()
      weights[i] += delete_weight
      i -= 1
    n -= 1
  allocation = ["disk" for i in range(snapshots)]
  for i in [p[0] for p in sorted(enumerate(weights), key = lambda p : p[1], reverse = True)][:snapshots_in_ram]:
    allocation[i] = "RAM"
    
  return weights, allocation
  
class MultistageManager:
  def __init__(self, max_n, snapshots_in_ram, snapshots_on_disk):
    if snapshots_in_ram == 0:
      storage = defaultdict(lambda : "disk")
    elif snapshots_on_disk == 0:
      storage = defaultdict(lambda : "RAM")
    else:
      storage = allocate_snapshots(max_n, snapshots_in_ram, snapshots_on_disk)[1]
    
    self._n = 0
    self._r = 0
    self._max_n = max_n
    self._snapshots_in_ram = snapshots_in_ram
    self._snapshots_on_disk = snapshots_on_disk
    self._snapshots = []
    self._storage = storage
    self._deferred_snapshot = None
    
  def n(self):
    return self._n
    
  def r(self):
    return self._r
    
  def max_n(self):
    return self._max_n
  
  def snapshots_in_ram(self):
    snapshots = 0
    for i in range(len(self._snapshots)):
      if self._storage[i] == "RAM":
        snapshots += 1
    return snapshots
  
  def snapshots_on_disk(self):
    snapshots = 0
    for i in range(len(self._snapshots)):
      if self._storage[i] == "disk":
        snapshots += 1
    return snapshots
  
  def snapshot(self):
    self._snapshots.append(self._n)
    self._deferred_snapshot = self._last_snapshot()
  
  def _last_snapshot(self):
    return self._snapshots[-1], self._storage[len(self._snapshots) - 1]
  
  def deferred_snapshot(self):
    deferred_snapshot, self._deferred_snapshot = self._deferred_snapshot, None
    return deferred_snapshot
  
  def forward(self):
    if self._n == self._max_n - self._r - 1:
      self._n += 1
    else:
      self._n += n_advance(self._max_n - self._n - self._r, self._snapshots_in_ram + self._snapshots_on_disk - len(self._snapshots) + 1)
  
  def reverse(self):
    self._r += 1
  
  def load_snapshot(self):
    self._n, storage = self._last_snapshot()
    delete = self._n == self._max_n - self._r - 1
    if delete:
      self._snapshots.pop()
    return self._n, storage, delete
