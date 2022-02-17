import argparse
import torch
from Algorithm import *

from Trainer import *
from Example import *
from Common.Utils import set_seed, gym_env

# policy_name = "policy_better"
policy_name = "QR_randomI2O/policy_best"

def hyperparameters():
    parser = argparse.ArgumentParser(description='Soft Actor Critic (SAC) v2 example')

    # develop mode
    parser.add_argument('--develop-mode', default=True, type=bool, help="you should check initial random on-off")

    # environment
    parser.add_argument('--algorithm', default='SAC_v2', type=str, help='you should choose same algorithm with loaded network')
    parser.add_argument('--domain-type', default='gym', type=str, help='gym or dmc, dmc/image')
    parser.add_argument('--env-name', default='QuadRate-v0', help='Pendulum-v0, MountainCarContinuous-v0')
    parser.add_argument('--discrete', default=True, type=bool, help='Always Continuous')
    parser.add_argument('--render', default=True, type=bool)
    parser.add_argument('--training-start', default=1000, type=int, help='First step to start training')
    parser.add_argument('--max-step', default=1000001, type=int, help='Maximum training step')
    parser.add_argument('--eval', default=True, type=bool, help='whether to perform evaluation')
    parser.add_argument('--eval-step', default=10000, type=int, help='Frequency in performance evaluation')
    parser.add_argument('--eval-episode', default=1, type=int, help='Number of episodes to perform evaluation')
    parser.add_argument('--random-seed', default=-1, type=int, help='Random seed setting')

    parser.add_argument('--cpu-only', default=False, type=bool, help='force to use cpu only')
    parser.add_argument('--log', default=False, type=bool, help='use tensorboard summary writer to log, if false, cannot use the features below')
    parser.add_argument('--tensorboard', default=True, type=bool, help='when logged, write in tensorboard')
    parser.add_argument('--file', default=False, type=bool, help='when logged, write log')
    parser.add_argument('--numpy', default=False, type=bool, help='when logged, save log in numpy')

    parser.add_argument('--model', default=False, type=bool, help='when logged, save model')
    parser.add_argument('--model-freq', default=10000, type=int, help='model saving frequency')
    parser.add_argument('--buffer', default=False, type=bool, help='when logged, save buffer')
    parser.add_argument('--buffer-freq', default=10000, type=int, help='buffer saving frequency')

    # save
    parser.add_argument('--path', default="X:/RMBRL/Results/", help='path for save')

    args = parser.parse_args()

    return args

def main(args_test):

    args = None

    if args_test.algorithm == 'SAC_v2':
        from Example.run_SACv2 import hyperparameters
    if args_test.algorithm == 'DDPG':
        from Example.run_SACv2 import hyperparameters

    args = hyperparameters()

    if args_test.cpu_only == True:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Device: ", device)
    # random seed setting
    random_seed = set_seed(args.random_seed)
    print("Random Seed:", random_seed)

    #env setting
    env, test_env = gym_env(args.env_name, random_seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]
    min_action = env.action_space.low[0]

    algorithm = None

    if args_test.algorithm == 'SAC_v2':
        algorithm = SAC_v2(state_dim, action_dim, device, args)

    elif args_test.algorithm == 'DDPG':
        algorithm = DDPG(state_dim, action_dim, device, args)

    algorithm.actor.load_state_dict(torch.load(args.path + policy_name))

    print("Training of", args.domain_type + '_' + args.env_name)
    print("Algorithm:", algorithm.name)
    print("State dim:", state_dim)
    print("Action dim:", action_dim)
    print("Max action:", max_action)
    print("Min action:", min_action)

    trainer = None

    if args_test.develop_mode is False:
        trainer = Basic_trainer(env, test_env, algorithm, max_action, min_action, args, args_test)
    else:
        trainer = MRC_trainer(env, test_env, algorithm, max_action, min_action, args, args_test)
    trainer.test()

if __name__ == '__main__':
    args_test = hyperparameters()
    main(args_test)