import numpy as np
from collections import deque
import gym

class FrameStack1D(gym.Wrapper):
  def __init__(self, env, k):
    gym.Wrapper.__init__(self, env)
    self.k = k
    self.frames = deque([], maxlen=k)
    shp = env.observation_space.shape[0]
    self.observation_space = gym.spaces.Box(low=0, high=1., shape=(shp, k), dtype=np.float32)

  def reset(self):
    ob = self.env.reset()
    for _ in range(self.k):
      self.frames.append(ob)
    return self._get_ob()

  def step(self, action):
    ob, reward, done, info = self.env.step(action)
    self.frames.append(ob)
    return self._get_ob(), reward, done, info

  def _get_ob(self):
    assert len(self.frames) == self.k
    return list(self.frames)
