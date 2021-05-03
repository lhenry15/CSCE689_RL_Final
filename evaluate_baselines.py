from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import time
from baseline_agents import RandomAgent, RuleAgent
from utils import evaluate
from env import make_env
import sys

DATAPATH = "data/"+sys.argv[1]+".csv"

if __name__ == "__main__":

    # Random Agent
    env = make_env([DATAPATH])
    for agent in [RandomAgent(), RuleAgent()]:
        mean_reward, std_reward = evaluate(env, agent)
        print('Agent:', agent.name, 'Mean:', mean_reward, 'Std:', std_reward)
