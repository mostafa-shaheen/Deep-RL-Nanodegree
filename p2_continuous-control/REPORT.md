# Report on Project 2: Continuous Control
This file reports the method adopted to train an agent for solving the task of continuous control of a double jointed arm in a Unity environment.

## Inputs
state space: 33 (Continuous)
action space: 4 (Continuous)
reward structure: +0.1 for reaching the goal position


##Method
The inputs are given to an agent which follows a policy for executing the task.
As the state space has real numbers, implementing a Q table is not possible. Hence, neural networks are used for function approximation.
Also, as the action space is continuous, methods such as DQN cannot be directly implemented.
An initial attempt with the REINFORCE method was made but it was found out that the training is unstable.
Therefore, the Deep Deterministic Policy Gradient (DDPG) algorithm which is an Actor-Critic method was used.
In this method the actor is the agent i.e. policy that takes the state and outputs actions.
While the Critic evaluates the expected values from a state,action pair i.e. Q value estimator

###The structure of the Actor is as follows:
- Fc1 = ReLU (input_state (states = 33) x 128 neurons).
- Fc2 = ReLU (F1 x 128 neurons).
- Fc3 = ReLU (F2 x output_state (actions = 4)).
###The structure of the Critic is as follows:
F1 = ReLU (input_state (states = 33) x 128 neurons)
F2 = ReLU (F1+action_size (=4) x 128 neurons)
F3 = ReLU (F2 x 1)
Two NNs for actor and critic of same architecture are used: local network (θ_local) and target network (θ_target).
The target network weights are soft updated using the local network weights.
                    **θ_target = τθ_local + (1 - τ)θ_target**
