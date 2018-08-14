from dataclasses import dataclass
import torch.nn as nn
from torch.nn import functional as F
from noisy import NoisyLinear

@dataclass
class DQN(nn.Module):
  states: int
  actions: int
  history: int
  atoms: int
  hidden: int = 32

  def __post_init__(self):
    super(DQN, self).__init__()
    self.fc1 = nn.Linear(self.states * self.history, self.hidden)
    self.fc2 = NoisyLinear(self.hidden, self.hidden)
    self.adv = NoisyLinear(self.hidden, self.actions * self.atoms)
    self.val = NoisyLinear(self.hidden, self.atoms)

  def forward(self, x, log=False):
    x = x.view(-1, self.states * self.history)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    adv = self.adv(x).view(-1, self.actions, self.atoms)
    val = self.val(x).view(-1, 1, self.atoms)
    q = val + adv - adv.mean(1, keepdim=True)
    # Use log softmax for numerical stability
    return F.log_softmax(q, dim=2) if log else F.softmax(q, dim=2)

  def sample_noise(self):
    self.fc2.sample_noise()
    self.adv.sample_noise()
    self.val.sample_noise()
