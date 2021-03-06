# Report on Project 2: Continuous Control
This file reports the method adopted to train an agent for solving the task of continuous control of a double jointed arm in a Unity environment.

## Inputs
- state space: 33 (Continuous).
- action space: 4 (Continuous).
- reward structure: +0.1 for reaching the goal position.


## Method
The inputs are given to an agent which follows a policy for executing the task.
As the state space has real numbers, implementing a Q table is not possible. Hence, neural networks are used for function approximation.
Also, as the action space is continuous, methods such as DQN cannot be directly implemented.
An initial attempt with the REINFORCE method was made but it was found out that the training is unstable.
Therefore, the Deep Deterministic Policy Gradient (DDPG) algorithm which is an Actor-Critic method was used.
In this method the actor is the agent i.e. policy that takes the state and outputs actions.
While the Critic evaluates the expected values from a state,action pair i.e. Q value estimator

### The structure of the Actor is as follows:

```python
Actor(
  (fc1): Linear(in_features=33, out_features=400, bias=True)
  (bn): BatchNorm1d(400, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fc2): Linear(in_features=400, out_features=300, bias=True)
  (fc3): Linear(in_features=300, out_features=4, bias=True)
)
```
### The structure of the Critic is as follows:

```python
Critic(
  (fcs1): Linear(in_features=33, out_features=400, bias=True)
  (bn): BatchNorm1d(400, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fc2): Linear(in_features=404, out_features=300, bias=True)
  (fc3): Linear(in_features=300, out_features=1, bias=True)
)
```
- Two NNs for actor and critic of same architecture are used: local network (θ_local) and target network (θ_target).
The target network weights are soft updated using the local network weights.
                    **θ_target = τθ_local + (1 - τ)θ_target**.
## Hyperparameters

```python
- BUFFER_SIZE = 1e6 # replay buffer size.
- BATCH_SIZE = 1024 # minibatch size.
- GAMMA = 0.92 # discount factor.
- TAU = 1e-3 # for soft update of target parameters.
- LR_ACTOR = 1e-4 # Actor Learning Rate.
- LR_CRITIC = 1e-3 # Critic Learning Rate.
- MAx_t =1000. #maximum number of timesteps per episode 
- WEIGHT_DECAY = 0 # L2 weight decay.
- UPDATE_EVERY = 20        # how often to update the network.
- UPDATE_NETWORK = 10      # update network this many times.
```
## Rewards plot
A plot of the average rewards received is seen below:
![scores_plot](https://github.com/mostafa-shaheen/Deep-RL-Nanodegree/blob/master/p2_continuous-control/score_plot.png)
- the environment was solved in 76 episodes as the next 100 episode scores were averaged to +30.0.
maybe it's more convincing to say that the environment was solved after the total 176 episodes because the agent actually was still learning during the last 100 episodes.

## Future ideas for improving agents performance
- Use a different Neural Network architecture for actor and critic.
- Implement with other methods such as A3C, PPO, D4PG for faster and improved agent performance.
- Implement using Hierarchical reinforcement learning.
  - Move near the goal roughly.
  - Reach goal fine.
