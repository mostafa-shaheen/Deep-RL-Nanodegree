#### Udacity Deep Reinforcement Learning Nanodegree
### Project 3: Multi-Agent Collaboration & Competition
# Train Two RL Agents to Play Tennis


## The Environment
We'll work with an environment that is similar, but not identical to the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment on the Unity ML-Agents GitHub page.

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to moves toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.
The environment is considered solved when the average (over 100 episodes) of those **scores** is at least +0.5.

## Goal
The goal of this project is to train two RL agents to play tennis. As in real tennis, the goal of each player is to keep the ball in play. And, when you have two equally matched opponents, you tend to see fairly long exchanges where the players hit the ball back and forth over the net.

## Approach

#### Multi-Agent Deep Deterministic Policy Gradient (MADDPG)
The original DDPG algorithm from which I extended to create the MADDPG version, is outlined in [this paper](https://arxiv.org/pdf/1509.02971.pdf), _Continuous Control with Deep Reinforcement Learning_, by researchers at Google Deepmind. In this paper, the authors present "a model-free, off-policy actor-critic algorithm using deep function approximators that can learn policies in high-dimensional, continuous action spaces." They highlight that DDPG can be viewed as an extension of Deep Q-learning to continuous tasks.

#### Actor-Critic Method
Actor-critic methods leverage the strengths of both policy-based and value-based methods.
Using a policy-based approach, the agent (actor) learns how to act by directly estimating the optimal policy and maximizing reward through gradient ascent. Meanwhile, employing a value-based approach, the agent (critic) learns how to estimate the value (i.e., the future cumulative reward) of different state-action pairs. Actor-critic methods combine these two approaches in order to accelerate the learning process. Actor-critic agents are also more stable than value-based agents, while requiring fewer training samples than policy-based agents.
## Networks architecture
#### Actor
~~~python
Actor(
  (fc1): Linear(in_features=24, out_features=256, bias=True)
  (bn): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fc2): Linear(in_features=256, out_features=128, bias=True)
  (fc3): Linear(in_features=128, out_features=2, bias=True)
)
~~~
#### Critic
~~~python
Critic(
  (fcs1): Linear(in_features=24, out_features=256, bias=True)
  (bn): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fc2): Linear(in_features=258, out_features=128, bias=True)
  (fc3): Linear(in_features=128, out_features=1, bias=True)
)
~~~

## Experience sharing
I have chosen to implement two separate networks(local & target) for each agent. thus, I was reaching a point where one agent keeps beating the other which also leads to ending episodes quickly so even the better agent wasn't able to learn more. to overcome this issue I thought of a method that chooses the most winner agent over the last 100 episodes as the best agent and copies it's models weights to the other. it did well! and the two agents kept improving together and reached an average score of over +1.0 over 100 consecutive episodes. 

## prioritized experience replay
Experience replay lets online reinforcement learning agents remember and reuse experiences from the past. In prior work, experience transitions were uniformly sampled from a replay memory. However, this approach simply replays transitions at the same frequency that they were originally experienced, regardless of their significance. prioritized experience replays important transitions more frequently, and therefore learns more efficiently. I used this [impelmentation](https://github.com/rlcode/per) that using a sum tree data structure which was recommended by project 2 reviewer.

## Hyperparameters

~~~python
BUFFER_SIZE = 2*int(1e5)  # replay buffer size
BATCH_SIZE = 256          # minibatch size
GAMMA = 0.96              # discount factor
TAU   = 1e-3              # for soft update of target parameters
LR_ACTOR  = 1e-4          # learning rate of the actor 
LR_CRITIC = 1e-3          # learning rate of the critic
WEIGHT_DECAY = 0          # L2 weight decay
UPDATE_EVERY = 2          # how often to update the network
UPDATE_NETWORK = 2        # how many times performing update
SHARE_EVERY = 50          # how many episodes to share weights from the better agent 
noise_ratio = 1/1000      # cancels noise after 1000 episodes
~~~

## Plot of Rewards
The image below is a plot of score(maximum over both agents) per episode to illustrate that the agents got an average score of 1.0 over 100 consecutive episodes.
(https://github.com/mostafa-shaheen/Deep-RL-Nanodegree/blob/master/p3_collaboration-competition/score_plot1.png)
the required average score to pass(+0.5) was achieved between episodes 900 and 950.
I decided to continue training the agents for another 1000 episodes to see if I can reach an average score of 1.5. 
but the score didn't exceed 1.06. Also, as shown below. it started oscillating and the instability had shown up in training like what was mentioned in the project lesson. 
(https://github.com/mostafa-shaheen/Deep-RL-Nanodegree/blob/master/p3_collaboration-competition/score_plot2.png)
## Ideas for Future Work

* Try solving the environment using the __Proximal Policy Optimization__ algorithm. A new family of policy gradient methods for reinforcement learning, which alternate between sampling data through interaction with the environment, and optimizing a "surrogate" objective function using stochastic gradient ascent. PPO outperforms other online policy gradient methods, and overall strikes a favorable balance between sample complexity, simplicity, and wall-time.
