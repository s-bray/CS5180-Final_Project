import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Gaussian policy network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, log_std_min=-20, log_std_max=2):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)

        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)

        self.max_action = max_action
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.randn_like(mean)
        x_t = mean + std * normal  # reparameterization trick
        y_t = torch.tanh(x_t)
        action = self.max_action * y_t

        # Log probability
        log_prob = -0.5 * ((x_t - mean) / std).pow(2) - log_std - 0.5 * np.log(2 * np.pi)
        log_prob = log_prob.sum(1, keepdim=True)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6).sum(1, keepdim=True)
        return action, log_prob


# Critic (Q1 and Q2 networks)
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Q1
        self.q1_l1 = nn.Linear(state_dim + action_dim, 256)
        self.q1_l2 = nn.Linear(256, 256)
        self.q1_out = nn.Linear(256, 1)

        # Q2
        self.q2_l1 = nn.Linear(state_dim + action_dim, 256)
        self.q2_l2 = nn.Linear(256, 256)
        self.q2_out = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.q1_l1(sa))
        q1 = F.relu(self.q1_l2(q1))
        q1 = self.q1_out(q1)

        q2 = F.relu(self.q2_l1(sa))
        q2 = F.relu(self.q2_l2(q2))
        q2 = self.q2_out(q2)

        return q1, q2


class SAC:
    def __init__(self, state_dim, action_dim, max_action, discount=0.99, tau=0.005, alpha=0.2, automatic_entropy_tuning=True, target_entropy=None):

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.total_it = 0

        # Entropy tuning
        self.automatic_entropy_tuning = automatic_entropy_tuning
        if self.automatic_entropy_tuning:
            if target_entropy is None:
                self.target_entropy = -action_dim  # heuristic
            else:
                self.target_entropy = target_entropy

            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)
            self.alpha = self.log_alpha.exp().detach()
        else:
            self.alpha = torch.tensor(alpha).to(device)


    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(device).unsqueeze(0)
        if evaluate:
            mean, _ = self.actor(state)
            action = torch.tanh(mean)
        else:
            action, _ = self.actor.sample(state)
        return action.cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # Sample next action
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_q = reward + not_done * self.discount * target_q

        # Get current Q estimates
        current_q1, current_q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        pi, log_pi = self.actor.sample(state)
        q1_pi, q2_pi = self.critic(state, pi)
        min_q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = (self.alpha * log_pi - min_q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Entropy coefficient update
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp().detach()
        else:
            alpha_loss = 0

        # Soft update target network
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")
        if self.automatic_entropy_tuning:
            torch.save(self.log_alpha, filename + "_log_alpha")
            torch.save(self.alpha_optimizer.state_dict(), filename + "_alpha_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        if self.automatic_entropy_tuning:
            self.log_alpha = torch.load(filename + "_log_alpha")
            self.alpha_optimizer.load_state_dict(torch.load(filename + "_alpha_optimizer"))
            self.alpha = self.log_alpha.exp().detach()

    def load_gpu2cpu(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic", map_location=torch.device('cpu')))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer", map_location=torch.device('cpu')))
        self.actor.load_state_dict(torch.load(filename + "_actor", map_location=torch.device('cpu')))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer", map_location=torch.device('cpu')))

        if self.automatic_entropy_tuning:
            self.log_alpha = torch.load(filename + "_log_alpha", map_location=torch.device('cpu'))
            self.alpha_optimizer.load_state_dict(torch.load(filename + "_alpha_optimizer", map_location=torch.device('cpu')))
            self.alpha = self.log_alpha.exp().detach()
