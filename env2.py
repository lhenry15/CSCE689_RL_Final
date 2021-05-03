# Wrapper of environment
import gym
from gym import spaces
from gym.utils import seeding
from stable_baselines import bench, logger

import numpy as np
import random

from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances
from sklearn.preprocessing import StandardScaler
import sys
from sklearn.neighbors import NearestNeighbors

from utils import read_data_as_matrix, run_iforest

class EnvBase(gym.Env):
    def __init__(self, datapath="data/toy.csv", budget=100, case=0):
        # Read dataset
        X_train, labels, anomalies = read_data_as_matrix(datapath)
        self.X_train = X_train
        self.labels = labels
        self.size = len(self.labels)
        self.budget = budget
        self.dim = X_train.shape[1]
        self.case = case
        print("Case:", self.case)
        if case == 0: # all feature
            self.state_dim = 8
        elif case == 1: # only feature 1 (if there is an anomaly in top-k neighbors)
            self.state_dim = 1
        elif case == 2: # score of unsupervised detector
            self.state_dim = 2
        elif case == 3 or case == 4: # only feature 2 (the min/mean/max distance to known anomalies)
            self.state_dim = 5

        # Unsupervised scores
        unsupervised_scores, self.feature_importances = run_iforest(self.X_train)
        self.scores = np.expand_dims(unsupervised_scores, axis=1)
        self.scores = (self.scores-np.mean(self.scores, axis=0)[None,:]) / np.std(self.scores, axis=0)[None,:]

        # Exatract distances features
        self.X_train = StandardScaler().fit_transform(self.X_train)
        self.NN = NearestNeighbors(n_neighbors=10)
        self.NN.fit(self.X_train)

        print("Data loaded: {} Total instances: {} Anomalies: {}".format(datapath, self.size, len(anomalies)))

        # Gym settings
        self.action_space = spaces.Discrete(2)
        high = np.ones(self.state_dim) * 10.0
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self):
        pass

class Env(EnvBase):

    def __init__(self, datapath, case):
        super().__init__(datapath=datapath, case=case)

    def step(self, action):
        """ Proceed to the next state given the curernt action
            1 for check the instance, 0 for not
            return next state, reward and done
        """
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        if action == 0:
            #r = -0.001
            r = 0.0
        elif action == 1:
            if self.labels[self.pointer] == 0: # pick anomaly
                r = 1
                self.anomalies.append(self.pointer)

            else: # pick normal data point
                #r = -0.05
                r = -0.01
                self.normalies.append(self.pointer)
            self.count += 1
        self.pointer += 1

        #sys.stdout.write("\r"+str(float(self.pointer)/self.size))
        #sys.stdout.flush()
        #print(self.pointer, self.size, 100*float(self.pointer)/self.size)

        if self.pointer >= self.size:
            self.done = True
        else:
            self.done = False

        return self._obs(), r, self.done, {}

    def reset(self):
        """ Reset the environment, for streaming evaluation
        """
        # Some stats
        self.pointer = 0
        self.count = 0
        self.done = False
        self.anomalies = []
        self.normalies = []
        self.labeled = []

        return self._obs()

    def _obs(self):
        """ Return the observation of the current state
        """
        if self.done:
            return np.zeros(self.state_dim)

        features = []
        curr_data_point = [self.X_train[self.pointer]]
        distances, near_neighbors = self.NN.kneighbors(curr_data_point, 10) 
        distances, near_neighbors = distances[0], near_neighbors[0]
        near_anomalies = np.where(np.isin(near_neighbors, self.anomalies), 1, 0)
        near_normalies = np.where(np.isin(near_neighbors, self.normalies), 1, 0)


        ## Avg distance to abnormal instances
        if len(self.anomalies) > 0:
            dists_to_a = euclidean_distances(curr_data_point, self.X_train[self.anomalies])
            a = np.mean(dists_to_a) 
            a_min = np.min(dists_to_a)
            a_max = np.max(dists_to_a)
        else:
            a = 0.0
            a_min = 0.0
            a_max = 99999

        ## Avg distance to normal instances
        if len(self.normalies) > 0:
            dists_to_n = euclidean_distances(curr_data_point, self.X_train[self.normalies])
            n = np.mean(dists_to_n)
            n_min = np.min(dists_to_n)
            n_max = np.max(dists_to_n)
        else:
            n = 0.0
            n_min = 0.0
            n_max = 99999

        c = self.scores[self.pointer]
        #importance_voting = np.zeros(self.dim)
        #for est in self.feature_importances:
        #    importance_voting[np.argmax(est)] += 1
        #features.extend(importance_voting)
        
        if self.case == 0:
            features.append(1) if np.count_nonzero(near_anomalies[:5])>0 else features.append(0)
            features.extend([a, a_min, a_max])
            features.extend([n, n_min, n_max])
            features.extend(c)

        elif self.case == 1:
            features.append(1) if np.count_nonzero(near_anomalies[:5])>0 else features.append(0)

        elif self.case == 2:
            features.append(1) if np.count_nonzero(near_anomalies[:5])>0 else features.append(0)
            features.extend(c)

        elif self.case == 3:
            features.append(1) if np.count_nonzero(near_anomalies[:5])>0 else features.append(0)
            features.extend([a, a_min, a_max])
            features.extend(c)

        elif self.case == 4:
            features.append(1) if np.count_nonzero(near_anomalies[:5])>0 else features.append(0)
            features.extend([n, n_min, n_max])
            features.extend(c)

        return features
        
class EnsembleEnv(gym.Env):
    def __init__(self, datapaths):
        self.envs = []
        for datapath in datapaths:
            self.envs.append(Env(datapath))

        # Gym settings
        self.action_space = self.envs[0].action_space
        self.observation_space = self.envs[0].observation_space
        self.seed()
        self.reset()

    def seed(self, seed=None):
        random.seed(seed)
        for env in self.envs:
            env.seed(seed)
        return [seed]

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        self.env = random.choice(self.envs)
        return self.env.reset()

    def render(self):
        pass

def make_env(datapaths, case=0):
    if len(datapaths) > 1:
        env = EnsembleEnv(datapaths, case=case)
    else:
        env = Env(datapaths[0], case=case)
    env = bench.Monitor(env, logger.get_dir())
    return env
