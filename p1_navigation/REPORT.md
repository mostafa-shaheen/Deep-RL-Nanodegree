# Report on project1-Navigation

In this project, the DQN learning algorithm has been used to solve the Navigation problem.
Other learning algorithms like Dobule DQN, Prioritized Experience Replay DQN, Dueling DQN will be added later.

The report will describe the learning algorithm with used hyper parameters, the arcitectures for neural netwoorks.

### Training Code
The code is written in PyTorch and Python3, executed in Jupyter Notebook
- Navigation.ipynb	: Main Instruction file
- dqn_agent.py	: Agent and ReplayBuffer Class
- model.py	: Build QNetwork and train function
- checkpoint.pth : Saved Model Weights

### Learning Algorithm
#### Deep Q-Network

**Q-learning** is a value-based Reinforcement Learning algorithm that is used to find the optimal action-selection policy using a q function, *`Q(s,a)`*

It's goal is to maximize the value function Q

<a href="https://www.codecogs.com/eqnedit.php?latex=\mathit{Q}^{*}(s,&space;a)&space;=&space;\underset{\pi}{\mathrm{max}}\left&space;\{&space;r_{t}&space;&plus;&space;\gamma&space;r_{t&plus;1}&space;&plus;&space;\gamma&space;^{2}&space;r_{t&plus;2}&plus;...|s_{t}=s,&space;a_{t}=a,&space;\pi&space;\right&space;\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathit{Q}^{*}(s,&space;a)&space;=&space;\underset{\pi}{\mathrm{max}}\left&space;\{&space;r_{t}&space;&plus;&space;\gamma&space;r_{t&plus;1}&space;&plus;&space;\gamma&space;^{2}&space;r_{t&plus;2}&plus;...|s_{t}=s,&space;a_{t}=a,&space;\pi&space;\right&space;\}" title="\mathit{Q}^{*}(s, a) = \underset{\pi}{\mathrm{max}}\left \{ r_{t} + \gamma r_{t+1} + \gamma ^{2} r_{t+2}+...|s_{t}=s, a_{t}=a, \pi \right \}" /></a>

which is the maximum sum of rewards r<sub>t</sub> discounted by &gamma; at each timestep t, achievable by a behaviour policy *&pi;=P(a|s)*, after making an
observation (s) and taking an action (a)

The follwoing is pseudo code of Q learning algorithm.
1. Initialze Q-values *Q(s,a)* arbitrarily for all state-action pairs.
2. For i=1 to # num_episodes <br/>
  Choose an action A<sub>t</sub> int eht current state (s) based on current Q-value estimates (e,g &epsilon;-greedy) </br>
  Take action A<sub>t</sub> amd observe reward and state, R<sub>t+1</sub>, S<sub>t+1</sub>
  Update *Q(s|a)* <br/>
  
    <a href="https://www.codecogs.com/eqnedit.php?latex=\mathit{Q(s_{t}|a_{t})}&space;=&space;\mathit{Q(s_{t}|a_{t})}&space;&plus;&space;\alpha(\mathitt{R_{t&plus;1}&plus;\gamma&space;\mathrm{max_{a}\mathit{Q(S_{t&plus;1},&space;a)-\mathit{Q(s_{t}|a_{t})}&space;}}})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathit{Q(s_{t}|a_{t})}&space;=&space;\mathit{Q(s_{t}|a_{t})}&space;&plus;&space;\alpha(\mathitt{R_{t&plus;1}&plus;\gamma&space;\mathrm{max_{a}\mathit{Q(S_{t&plus;1},&space;a)-\mathit{Q(s_{t}|a_{t})}&space;}}})" title="\mathit{Q(s_{t}|a_{t})} = \mathit{Q(s_{t}|a_{t})} + \alpha(\mathitt{R_{t+1}+\gamma \mathrm{max_{a}\mathit{Q(S_{t+1}, a)-\mathit{Q(s_{t}|a_{t})} }}})" /></a>

**Q-networks** approximate the Q-function as a neural network given a state, Q-values for each action<br/>
*Q(s, a, θ)* is a neural network that define obejctive function by mean-squared error in Q-values
  <a href="https://www.codecogs.com/eqnedit.php?latex=\mathfrak{L}{(\theta)&space;=&space;\mathrm{E}\left&space;[&space;\left&space;(&space;\underbrace{r&space;&plus;&space;\gamma&space;\underset{a^{'}}{\mathrm{max}}Q(s^{'},a^{'},\theta)}&space;-&space;Q(s,a,\theta)\right&space;)^{2}&space;\right&space;]&space;}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathfrak{L}{(\theta)&space;=&space;\mathrm{E}\left&space;[&space;\left&space;(&space;\underbrace{r&space;&plus;&space;\gamma&space;\underset{a^{'}}{\mathrm{max}}Q(s^{'},a^{'},\theta)}&space;-&space;Q(s,a,\theta)\right&space;)^{2}&space;\right&space;]&space;}" title="\mathfrak{L}{(\theta) = \mathrm{E}\left [ \left ( \underbrace{r + \gamma \underset{a^{'}}{\mathrm{max}}Q(s^{'},a^{'},\theta)} - Q(s,a,\theta)\right )^{2} \right ] }" /></a>
  <br/>

To find optimum parameters &theta;, optimise by SGD, using &delta;*L(&theta;)*/&delta;*&theta;* <br/>
This algorithm diverges because stages are correlated and targets are non-stationary. 

**DQN-Experience replay**<br/>
In order to deal with the correlated states, the agent build a dataset of experience and then makes random samples from
the dataset.<br/>

<a href="https://www.codecogs.com/eqnedit.php?latex=\mathfrak{L}{(\theta)&space;=&space;\mathrm{E_{\mathit{s,a,r,s^{'}&space;D}}}\left&space;[&space;\left&space;(r&space;&plus;&space;\gamma&space;\underset{a^{'}}{\mathrm{max}}Q(s^{'},a^{'},\theta)&space;-&space;Q(s,a,\theta)\right&space;)^{2}&space;\right&space;]&space;}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathfrak{L}{(\theta)&space;=&space;\mathrm{E_{\mathit{s,a,r,s^{'}&space;D}}}\left&space;[&space;\left&space;(r&space;&plus;&space;\gamma&space;\underset{a^{'}}{\mathrm{max}}Q(s^{'},a^{'},\theta)&space;-&space;Q(s,a,\theta)\right&space;)^{2}&space;\right&space;]&space;}" title="\mathfrak{L}{(\theta) = \mathrm{E_{\mathit{s,a,r,s^{'} D}}}\left [ \left (r + \gamma \underset{a^{'}}{\mathrm{max}}Q(s^{'},a^{'},\theta) - Q(s,a,\theta)\right )^{2} \right ] }" /></a>

**DQN-Fixed Target** <br/>
Also, the agent fixes the parameter &theta;<sup>-</sup> and then with some frequency updates them<br/>

<a href="https://www.codecogs.com/eqnedit.php?latex=\mathfrak{L}{(\theta)&space;=&space;\mathrm{E_{\mathit{s,a,r,s^{'}&space;D}}}\left&space;[&space;\left&space;(r&space;&plus;&space;\gamma&space;\underset{a^{'}}{\mathrm{max}}Q(s^{'},a^{'},\theta^-)&space;-&space;Q(s,a,\theta)\right&space;)^{2}&space;\right&space;]&space;}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathfrak{L}{(\theta)&space;=&space;\mathrm{E_{\mathit{s,a,r,s^{'}&space;D}}}\left&space;[&space;\left&space;(r&space;&plus;&space;\gamma&space;\underset{a^{'}}{\mathrm{max}}Q(s^{'},a^{'},\theta^-)&space;-&space;Q(s,a,\theta)\right&space;)^{2}&space;\right&space;]&space;}" title="\mathfrak{L}{(\theta) = \mathrm{E_{\mathit{s,a,r,s^{'} D}}}\left [ \left (r + \gamma \underset{a^{'}}{\mathrm{max}}Q(s^{'},a^{'},\theta^-) - Q(s,a,\theta)\right )^{2} \right ] }" /></a>


**Neural Network Architecture**<br/>
The state space has 37 dimensions and the size of action space is 4.<br/>
so the number of input features of Neural Network is 37 and the output size is 4.<br/>
And the number of hidden layers and each size is configurable in this project.<br/>
You can input the list of hidden layers as one of the input parameters when creating an agent.<br/>
The hidden layers used in this project is [64,64] ie, 2 layers with 64, 64 neurons in each layer. <br/>

Network dimensions:
* Input layers  : 37
* Hidden layer1 : 64
* Hidden layer2 : 64
* Output layer  : 4

~~~python
QNetwork(
  (fc1): Linear(in_features=37, out_features=64, bias=True)
  (fc2): Linear(in_features=64, out_features=64, bias=True)
  (fc3): Linear(in_features=64, out_features=4, bias=True)
)
~~~

### Hyperparameters
Parameters that showed the best results are:
- `BUFFER_SIZE` = 1e6 , recommended in the dqn paper
- `BATCH_SIZE`  = 64 ,  minibatch size
- `GAMMA`       = 0.99 , discount factor
- `TAU`         = 1e-3 , parameter for soft update of target parameters
- `LR`          = 5e-4 , learning rate
- `UPDATE_EVERY`= 4, how often to update the network
- `EPS_DECAY` = 0.995, how much to decay epsilon from 1.0 to 0.01, for epsilon-greedy action selection


### Plot of Rewards

A plot of rewards per episode
- plot an average reward (over 100 episodes)
- It shows this agent solve the environment in in 404 episodes!
![image](https://github.com/mostafa-shaheen/Deep-RL-Nanodegree/blob/master/p1_navigation/average_scores.png "DQN")
![image](https://github.com/mostafa-shaheen/Deep-RL-Nanodegree/blob/master/p1_navigation/Score_per_episode_plot.png "DQN")

## Improvements

- Since Deep Q-Learning tends to overestimate action values, **Double DQN** has been shown as a good solution for that. 
Instead of blindly trusting the max value of Q-values that are completely random at the beginning of learning, we select the best action using one set of parameters w and evaluate it using a different set of parameters w'. It is like having two separate function approximators that must agree on the best action. The second set of parameters are the ones from target network that are frozen for some time and so different enough to be reused for this purpose. In the long run, this prevents the algorithm from propagating incidental high rewards that may have been obtained by chance, and do not reflect long-term returns. 
- Another improvement can be **Prioritized Experience Replay** which is based on the idea that the agent can learn more effectively from some transitions than from others, and the more important transitions should be sampled with higher probability. The importance of a sample is measured with the TD error, where the bigger the error, the more we expect to learn from that tuple. 
