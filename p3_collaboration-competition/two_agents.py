import numpy as np
import random
import copy
from collections import namedtuple, deque

from models import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim
from prioritized_memory import Memory

Num_agents = 2
BUFFER_SIZE = 2*int(1e5)  # replay buffer size
BATCH_SIZE = 256        # minibatch size
GAMMA = 0.96            # discount factor
TAU   = 1e-3              # for soft update of target parameters
LR_ACTOR  = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
UPDATE_EVERY = 2        # how often to update the network
UPDATE_NETWORK = 2




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agents():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, random_seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size  (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actors           = [Actor(state_size, action_size, random_seed).to(device) for i in range(Num_agents)]
        self.actors_target    = [Actor(state_size, action_size, random_seed).to(device) for i in range(Num_agents)]
        self.actors_optimizer = [optim.Adam(self.actors[i].parameters(), lr=LR_ACTOR)   for i in range(Num_agents)]

        # Critic Network (w/ Target Network)
        self.critics           = [Critic(state_size, action_size, random_seed).to(device) for i in range(Num_agents)]
        self.critics_target    = [Critic(state_size, action_size, random_seed).to(device) for i in range(Num_agents)]
        self.critics_optimizer = [optim.Adam(self.critics[i].parameters(), lr=LR_CRITIC, 
                                                               weight_decay=WEIGHT_DECAY) for i in range(Num_agents)]

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        #self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        self.prioritized_memory = Memory(BUFFER_SIZE)
        self.t_step = 0

    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience and it's priority
        for i in range(len(states)):
            
                #self.memory.add(states[i], actions[i], rewards[i], next_states[i], dones[i])

                states_tensor      = torch.from_numpy(states[i]).float().unsqueeze(0).to(device)
                next_states_tensor = torch.from_numpy(next_states[i]).float().unsqueeze(0).to(device)
                actions_tensor     = torch.from_numpy(actions[i]).float().unsqueeze(0).to(device)
                
                self.actors_target[i].eval()
                self.critics_target[i].eval()
                self.critics[i].eval()
                
                actions_next   = self.actors_target[i](next_states_tensor)
                Q_targets_next = self.critics_target[i](next_states_tensor, actions_next)
                Q_targets = rewards[i] + (GAMMA * Q_targets_next.detach() * (1 - dones[i]))
                Q_expected = self.critics[i](states_tensor, actions_tensor)
                #critic_loss = F.mse_loss(Q_expected, Q_targets)
                
                self.actors_target[i].train()
                self.critics_target[i].train()
                self.critics[i].train()
                
                error = abs((Q_expected - Q_targets).item())
                self.prioritized_memory.add(error, (states[i], actions[i], rewards[i], next_states[i], dones[i]))
                
        # Learn, if enough samples are available in memory
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            if self.prioritized_memory.tree.n_entries > BATCH_SIZE:
                for i in range(UPDATE_NETWORK):
                    #experiences = [self.memory.sample() for i in range(Num_agents)]
                    #self.learn(experiences, GAMMA)
                    self.learn(GAMMA)

    def act(self, states, noise=0):
        """Returns actions for given state as per current policy."""
        actions = np.zeros((2,2))
        states = torch.from_numpy(states).float().to(device)
        
        for i,state in enumerate(states):
            self.actors[i].eval()
            with torch.no_grad():
                actions[i] = self.actors[i](state.unsqueeze(0)).cpu().data.numpy()
            self.actors[i].train()

            if noise:
                actions += (self.noise.sample()*noise)

        return np.clip(actions, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        for i in range(Num_agents):
            
            #states, actions, rewards, next_states, dones = experiences[i]
            
            mini_batch, idxs, is_weights = self.prioritized_memory.sample(BATCH_SIZE)
            states, actions, rewards, next_states, dones = mini_batch

            # ---------------------------- update critic ---------------------------- #
            # Get predicted next-state actions and Q values from target models
            actions_next   = self.actors_target[i](next_states)
            Q_targets_next = self.critics_target[i](next_states, actions_next)
            # Compute Q targets for current states (y_i)
            Q_targets = rewards + (gamma * Q_targets_next.detach() * (1 - dones))
            # Compute critic loss
            Q_expected = self.critics[i](states, actions)

            # update priority
            errors = torch.abs(Q_expected - Q_targets).data.cpu().numpy()
            for j in range(BATCH_SIZE):
                idx = idxs[j]
                self.prioritized_memory.update(idx, errors[j])
                
            critic_loss = (torch.FloatTensor(is_weights).to(device) * F.mse_loss(Q_expected, Q_targets)).mean()
            # Minimize the loss
            self.critics_optimizer[i].zero_grad()
            critic_loss.backward()
            #torch.nn.utils.clip_grad_norm_(.parameters(), 1)
            self.critics_optimizer[i].step()

            # ---------------------------- update actor ---------------------------- #
            # Compute actor loss
            actions_pred = self.actors[i](states)
            actor_loss = -self.critics[i](states, actions_pred).mean()
            # Minimize the loss
            self.actors_optimizer[i].zero_grad()
            actor_loss.backward()
            self.actors_optimizer[i].step()

            # ----------------------- update target networks ----------------------- #
            self.soft_update(self.critics[i], self.critics_target[i], TAU)
            self.soft_update(self.actors[i] , self.actors_target[i], TAU)                     

            
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
    def share_experince(self, winner):
        for i in range(Num_agents):
            if i != winner:
                
                self.actors[i]  = self.actors[winner]
                self.critics[i] = self.critics[winner]
                
                self.actors_target[i]  = self.actors_target[winner]
                self.critics_target[i] = self.critics_target[winner]
                
                print('\n',winner,' shared to the other agent')
                

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.1):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state
