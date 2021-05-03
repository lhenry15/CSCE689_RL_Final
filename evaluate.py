from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

import argparse
import os
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from env2 import make_env
from stable_baselines import PPO2
from stable_baselines.common.evaluation import evaluate_policy

def argsparser():
    parser = argparse.ArgumentParser("Active Anomaly Detection")
    parser.add_argument('--test', help='Testing datasets', default='shuttle')
    parser.add_argument('--load', help='the model directory', default='log/shuttle_c0/model.zip')
    parser.add_argument('--case', help='case studies', type=int, default=0)

    return parser
    
def evaluate(args):

    test_datasets = args.test.split(',')

    # Generate the paths of datasets
    datapaths = []
    for d in test_datasets:
        datapaths.append(os.path.join('./data', d+'.csv'))

    # Make the testing environments
    env = make_env(datapaths, case=args.case)

    # Load model
    model = PPO2.load(args.load)

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=False)

    print('Mean:', mean_reward, 'Std:', std_reward)

if __name__ == "__main__":
    parser = argsparser()
    args = parser.parse_args()
    evaluate(args)

