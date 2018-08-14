from dataclasses import dataclass
import random
import torch
import torch.nn as nn
import torch.optim as optim
from model import DQN

@dataclass
class Agent:
  state: int
  actions: int
  history: int = 4
  atoms: int = 5 #51
  Vmin: float = -10
  Vmax: float = 10
  
  lr: float = 1e-5
  batch_size: int = 32
  discount: float = 0.99
  norm_clip: float = 10.

  def __post_init__(self):
    self.support = torch.linspace(self.Vmin, self.Vmax, self.atoms)
    self.delta_z = (self.Vmax - self.Vmin) / (self.atoms - 1)

    self.online_net = DQN(self.state, self.actions, self.history, self.atoms)
    self.online_net.train()

    self.target_net = DQN(self.state, self.actions, self.history, self.atoms)
    self.update_target_net()
    self.target_net.train()
    for param in self.target_net.parameters(): param.requires_grad = False

    self.optimiser = optim.Adam(self.online_net.parameters(), lr=self.lr)

  def act(self, state):
    state = torch.FloatTensor(state).unsqueeze(0)
    with torch.no_grad():
      return (self.online_net(state) * self.support).sum(2).argmax(1).item()

  def act_e_greedy(self, state, epsilon=0.001):
    return random.randrange(self.actions) if random.random() < epsilon else self.act(state)

  def learn(self, buffer):
    state, action, reward, next_state, terminal, weights, idx = buffer.sample(self.batch_size)
    state = torch.FloatTensor(state)
    action = torch.LongTensor(action)
    reward = torch.FloatTensor(reward)
    next_state = torch.FloatTensor(next_state)
    terminal = torch.FloatTensor(terminal)
    weights = torch.FloatTensor(weights)

    log_ps = self.online_net(state, log=True)
    log_ps_a = log_ps[range(self.batch_size), action]

    with torch.no_grad():
      # Calculate nth next state probabilities
      pns = self.online_net(next_state)
      dns = self.support.expand_as(pns) * pns
      argmax_indices_ns = dns.sum(2).argmax(1)
      self.target_net.sample_noise()
      pns = self.target_net(next_state)
      pns_a = pns[range(self.batch_size), argmax_indices_ns]

      # Compute Bellman operator T applied to z
      Tz = reward.unsqueeze(1) + (1 - terminal).unsqueeze(1) * self.discount * self.support.unsqueeze(0) # -10 ... 10 + reward
      Tz.clamp_(min=self.Vmin, max=self.Vmax)
      
      # Compute L2 projection of Tz onto fixed support z
      b = (Tz - self.Vmin) / self.delta_z # 0 ... 4
      l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
      # Fix disappearing probability mass when l = b = u (b is int)
      l[(u > 0) * (l == u)] -= 1
      u[(l < (self.atoms - 1)) * (l == u)] += 1

      # Distribute probability of Tz
      m = state.new_zeros(self.batch_size, self.atoms)
      offset = torch.linspace(0, ((self.batch_size - 1) * self.atoms), self.batch_size).unsqueeze(1).expand(self.batch_size, self.atoms).to(action)
      m.view(-1).index_add_(0, (l + offset).view(-1), (pns_a * (u.float() - b)).view(-1))  # m_l = m_l + p(s_t+n, a*)(u - b)
      m.view(-1).index_add_(0, (u + offset).view(-1), (pns_a * (b - l.float())).view(-1))  # m_u = m_u + p(s_t+n, a*)(b - l)

    loss = -torch.sum(m * log_ps_a, 1)  # Cross-entropy loss (minimises DKL(m||p(s_t, a_t)))
    loss = weights * loss

#     q_values = self.online_net(state)
#     q_value = q_values[range(self.batch_size), action]

#     next_q_values = self.target_net(next_state)
#     next_q_value = next_q_values.max(1)[0]

#     expected_q_value = reward + self.discount * next_q_value * (1 - terminal)
#     loss = weights * (q_value - expected_q_value).pow(2)

    self.optimiser.zero_grad()
    loss.mean().backward()
    self.optimiser.step()
    nn.utils.clip_grad_norm_(self.online_net.parameters(), self.norm_clip)

    buffer.update_priorities(idx, loss.tolist())

  def update_target_net(self):
    self.target_net.load_state_dict(self.online_net.state_dict())

  def sample_noise(self):
    self.online_net.sample_noise()

  def save(self, path):
    torch.save(self.online_net.state_dict(), path)

  # Evaluates Q-value based on single state (no batch)
  def evaluate_q(self, state):
    with torch.no_grad():
      return self.online_net(state.unsqueeze(0)).max(1)[0].item()

  def train(self):
    self.online_net.train()

  def eval(self):
    self.online_net.eval()
