import numpy as np

class BaseAgent(object):
    def __init__(self):
        pass

    def step(self, s, legal):
        pass
        
class RandomAgent(BaseAgent):
    def __init__(self):
        self.name = 'RandomAgent'

    def step(self, s):
        action = np.random.choice(2, p=[0.5,0.5])
        return action 

class RuleAgent(BaseAgent):

    def __init__(self):
        self.name = 'RuleAgent'

    def step(self, s):
        action = np.random.randint(2)
        near_anomalies, a, a_min, a_max, n, n_min, n_max, score = s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]
        if a_max > n_min or near_anomalies == 1:
            action = np.random.choice(2, p=[0.9, 0.1])
        else:
            action = np.random.choice(2, p=[0.1, 0.9])
        
        return action 

#class GreedyAgent(BaseAgent):
#
#    def __init__(self, env):
#        self.env = env 
#        self.env_nS = self.env.observation_space
#        self.env_nA = 2
#        self.policy = np.ones([self.env.nS, self.env.nA]) / self.env.nA
#
#    def step(self, s):
#        prev = self.env.scores
#        action = np.random.randint(2)
#
#        self.train_episode()
#        
#        return np.random.randint(2)
#
#    def policy_eval(self):
#        T = np.zeros([self.env_nS, self.env_nS])
#        R = np.zeros(self.env_nS)
#        for s1
#
#    def train_episode(self):
        

