import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        epsilon =  0.0009
        self.gamma = 1
        self.alpha = 0.01
        self.probs   = [1-epsilon+(epsilon/6), epsilon/6, epsilon/6, epsilon/6,  epsilon/6,  epsilon/6]
    
    def get_next_action_set_orderd(self, state):
        greedy_action = np.argmax(self.Q[state])
        action_set = np.append(greedy_action,np.delete(np.arange(6),greedy_action))
        return action_set

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        greedy_action = np.argmax(self.Q[state])
        action_set = np.append(greedy_action,np.delete(np.arange(6),greedy_action))
        action = np.random.choice(action_set, p=self.probs)
        return  action

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        
        next_action_set    = self.get_next_action_set_orderd(next_state)
        next_q_values      = np.array([self.Q[next_state][a] for a in next_action_set])
        Gt                 = reward + self.gamma *sum(next_q_values*self.probs)
        self.Q[state][action] = (1-self.alpha)*self.Q[state][action] + self.alpha*Gt