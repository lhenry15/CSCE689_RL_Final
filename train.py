from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

import argparse
import os
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from stable_baselines import logger

from env2 import make_env
from stable_baselines import PPO2

def argsparser():
    parser = argparse.ArgumentParser("Active Anomaly Detection")
    parser.add_argument('--train', help='Training datasets', default='shuttle')
    parser.add_argument('--test', help='Testing datasets', default='shuttle')
    parser.add_argument('--num_timesteps', help='The number of timesteps', type=int, default=10000)
    parser.add_argument('--log', help='the directory to save logs and models', default='log')
    parser.add_argument('--log_interval', help='the interval of log', type=int, default=10)
    parser.add_argument('--eval_log_interval', help='the interval of evaluation log on testing datasets', type=int, default=100)
    parser.add_argument('--case', help='state representation abalation study', type=int, default=0)

    return parser
    
def train(args):

    train_datasets = args.train.split(',')

    # Generate the paths of datasets
    datapaths = []
    for d in train_datasets:
        datapaths.append(os.path.join('./data', d+'.csv'))

    # Make the training environment
    env = make_env(datapaths, args.case)
 
    # Train the model
    model = PPO2('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=args.num_timesteps, log_interval=args.log_interval)
    model.save(os.path.join(args.log, 'model'))

if __name__ == "__main__":
    parser = argsparser()
    args = parser.parse_args()
    logger.configure(args.log)
    train(args)

