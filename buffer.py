import random
from collections import namedtuple, deque
from dataclasses import dataclass
from segment_tree import SumDeque
import numpy as np
import ipdb

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'terminal'))

@dataclass
class ReplayBuffer(object):
  capacity: int
  multi_step: int = 1 # relative to history, i.e. if ms=2, h=3: state=(1,2,3), next=(5,6,7)
  history: int = 1
  alpha: float = .5 # how much prioritization is used - 0 off, 1 full
  beta: float = .4
  
  def __post_init__(self):
    self._buffer = deque(maxlen=self.capacity)
    self._prior = SumDeque(self.capacity)
        
  def append(self, t: Transition):
    self._buffer.append(t)
    self._prior.append(0) # new elements with 0 prior - they cannot be sampled, yet they do not have next state
    if self._idx_max > self._idx_min: # but later they move in que to allowed region and get max prior
      self._prior[self._idx_max] = max(self._prior.max, 1.0)
    
    if len(self) > self._idx_min + 1: # also first elements cannot be sampled,
      for i in range(self._idx_min):  # because they do not have history states,
        self._prior[i] = 0            # they also get 0 prior

  def _get_state(self, idx, is_next):
    result, end_state = [], None
    loop_dir = 1 if is_next else -1
    lookup = 0 if is_next else -1
    # next: 1 2 3 5 5 5 5
    # cur:  1 1 1 2 3 4 5
    for step in range(self.history):
      if end_state is None:
        i = idx + step * loop_dir
        s = self._buffer[i].state
        if self._buffer[i + lookup].terminal: end_state = s
        result.append(s)
      else:
        result.append(end_state)
    return result[::loop_dir]

  @property
  def _idx_min(self): return self.history - 1
  @property
  def _idx_max(self): return len(self) - self.multi_step - self.history
  
  def _sample_idx(self, batch_size):
    res = []
    while len(res) < batch_size:
      idx = self._prior.find_prefixsum_idx(random.random() * self._prior.sum())
      assert self._idx_min <= idx <= self._idx_max
      res.append(idx)
    return res
  
  def _get_transition(self, idx):
    state, next_state, action, reward, terminal = [], [], [], [], []
    for i in idx:
      if self._buffer[i].terminal:
        ns = [self._buffer[i].state for h in range(self.history)]
      else:
        ns = self._get_state(i + self.multi_step, True)
      next_state.append(ns)
      state.append(self._get_state(i, False))
      action.append(self._buffer[i].action)
      reward.append(self._buffer[i].reward)
      terminal.append(self._buffer[i].terminal)
    return state, action, reward, next_state, terminal
  
  def _get_weights(self, idx):
    probs = np.array([self._prior[i] / self._prior.sum() for i in idx])
    weights = (probs * len(self)) ** -self.beta
    return weights / weights.max()
    
  def sample(self, batch_size):
    idx = self._sample_idx(batch_size)
    return self._get_transition(idx) + (self._get_weights(idx), idx)

  def update_priorities(self, idxes, priorities):
    for idx, priority in zip(idxes, priorities):
      self._prior[idx] = priority ** self.alpha

  def can_sample(self, batch_size):
    return self._idx_max - self._idx_min > batch_size
    
  def __len__(self):
    return len(self._buffer)