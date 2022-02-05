import math, random, time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from Common.Utils import *
from Common.ControlledAR import ControlledAR

from gym import utils
from gym.envs.mujoco import mujoco_env

from gym.envs.mujoco.quad_rate import QuadRateEnv

class Car_trainer():
    def __init__(self, env, test_env, algorithm, max_action, min_action, args, args_test=None):

        if args_test is None:
            self.args = args
        else:
            self.args = args_test

        self.domain_type = self.args.domain_type
        self.env_name = self.args.env_name
        self.env = env
        self.test_env = test_env

        self.algorithm = algorithm

        self.max_action = max_action
        self.min_action = min_action

        self.discrete = self.args.discrete
        self.render = self.args.render
        self.max_step = self.args.max_step

        self.eval = self.args.eval
        self.eval_episode = self.args.eval_episode
        self.eval_step = self.args.eval_step

        self.episode = 0
        self.episode_reward = 0
        self.total_step = 0
        self.local_step = 0
        self.eval_num = 0

        self.train_mode = None

        self.path = self.args.path

        # score
        self.score = 0
        self.robust_score = 0
        self.total_score = 0
        self.best_score = 0

        if args.train_mode == 'offline':
            self.train_mode = self.offline_train
        elif args.train_mode == 'online':
            self.train_mode = self.online_train
        elif args.train_mode == 'batch':
            self.train_mode = self.batch_train

        assert self.train_mode is not None

        self.log = self.args.log

        self.model = self.args.model
        self.model_freq = self.args.model_freq
        self.buffer = self.args.buffer
        self.buffer_freq = self.args.buffer_freq

        # for develop
        self.env_sim = QuadRateEnv()
        self.env_real = QuadRateEnv()

        self.car = ControlledAR(self.env_sim, self.env_real, self.args)

    def offline_train(self, d, local_step):
        if d:
            return True
        return False

    def online_train(self, d, local_step):
        return True

    def batch_train(self, d, local_step):#VPG, TRPO, PPO only
        if d or local_step == self.algorithm.batch_size:
            return True
        return False

    def evaluate(self):
        self.eval_num += 1
        episode = 0
        reward_list = []
        alive_cnt = 0

        while True:
            self.local_step = 0
            if episode >= self.eval_episode:
                break
            episode += 1
            eval_reward = 0
            observation = self.test_env.reset()

            if '-ram-' in self.env_name:  # Atari Ram state
                observation = observation / 255.

            done = False

            while not done:
                self.local_step += 1
                action = self.algorithm.eval_action(observation)

                if self.discrete == False:
                    env_action = self.max_action * np.clip(action, -1, 1)
                else:
                    env_action = action

                next_observation, reward, done, _ = self.test_env.step(env_action)

                if self.render == True:
                    if self.domain_type in {'gym', "atari"}:
                        self.test_env.render()
                    elif self.domain_type in {'procgen'}:
                        cv2.imshow("{}_{}_{}".format(self.algorithm.name, self.domain_type, self.env_name), self.env.render(mode='rgb_array'))
                        cv2.waitKey(1)
                    elif self.domain_type in {'dmc', 'dmcr'}:
                        cv2.imshow("{}_{}_{}".format(self.algorithm.name, self.domain_type, self.env_name), self.env.render(mode='rgb_array', height=240, width=320))
                        cv2.waitKey(1)

                eval_reward += reward
                observation = next_observation

                if self.local_step == 8000:
                    alive_cnt += 1

            reward_list.append(eval_reward)
        score_now = sum(reward_list) / len(reward_list)
        alive_rate = alive_cnt / self.eval_episode

        if score_now > self.score:
            torch.save(self.algorithm.actor.state_dict(),
                       self.path + "policy_better")
            self.score = score_now
        if alive_rate > 0.9:
            torch.save(self.algorithm.actor.state_dict(),
                       self.path + "policy_current")
        if alive_cnt != 0 and score_now*alive_rate > self.total_score:
            torch.save(self.algorithm.actor.state_dict(),
                       self.path + "policy_total")
            self.total_score = score_now*alive_rate
        if alive_rate >= 0.8 and score_now*alive_rate > self.best_score:
            torch.save(self.algorithm.actor.state_dict(),
                       self.path + "policy_best")
            self.best_score = score_now*alive_rate

        print("Eval  | Average Reward {:.2f}, Max reward: {:.2f}, Min reward: {:.2f}, Stddev reward: {:.2f}, alive rate : {:.2f}".format(sum(reward_list)/len(reward_list), max(reward_list), min(reward_list), np.std(reward_list), 100*alive_rate))
        self.test_env.close()

    def run(self):
        while True:
            if self.total_step > self.max_step:
                print("Training finished")
                break

            self.episode += 1
            self.episode_reward = 0
            self.local_step = 0

            observation = self.env.reset()
            done = False

            while not done:
                self.local_step += 1
                self.total_step += 1

                if self.render == True:
                    if self.domain_type in {'gym', "atari"}:
                        self.env.render()
                    elif self.domain_type in {'procgen'}:
                        cv2.imshow("{}_{}_{}".format(self.algorithm.name, self.domain_type, self.env_name), self.env.render(mode='rgb_array'))
                        cv2.waitKey(1)
                    elif self.domain_type in {'dmc', 'dmcr'}:
                        cv2.imshow("{}_{}_{}".format(self.algorithm.name, self.domain_type, self.env_name), self.env.render(mode='rgb_array', height=240, width=320))
                        cv2.waitKey(1)

                if '-ram-' in self.env_name:  # Atari Ram state
                    observation = observation / 255.

                if self.total_step <= self.algorithm.training_start:
                   action = self.env.action_space.sample()
                   next_observation, reward, done, _ = self.env.step(action)
                else:
                    if self.algorithm.buffer.on_policy == False:
                        action = self.algorithm.get_action(observation)
                    else:
                        action, log_prob = self.algorithm.get_action(observation)

                    if self.discrete == False:
                        env_action = self.max_action * np.clip(action, -1, 1)
                    else:
                        env_action = action

                    next_observation, reward, done, info = self.env.step(env_action)
                if self.local_step + 1 == 1000:
                    real_done = 0.
                else:
                    real_done = float(done)

                self.episode_reward += reward

                if self.env_name == 'Pendulum':
                    reward = (reward + 8.1368) / 8.1368

                if self.algorithm.buffer.on_policy == False:
                    self.algorithm.buffer.add(observation, action, reward, next_observation, real_done)
                else:
                    self.algorithm.buffer.add(observation, action, reward, next_observation, real_done, log_prob)

                observation = next_observation

                if self.total_step >= self.algorithm.training_start and self.train_mode(done, self.local_step):
                    loss_list = self.algorithm.train(self.algorithm.training_step)

                if self.eval == True and self.total_step % self.eval_step == 0:
                    self.evaluate()

            # print("alpha : ", self.algorithm.log_alpha.exp())
            print("Train | Episode: {}, Reward: {:.2f}, Local_step: {}, Total_step: {}".format(self.episode, self.episode_reward, self.local_step, self.total_step))
        self.env.close()

    def test(self):
        self.eval_num += 1
        episode = 0
        reward_list = []
        alive_cnt = 0

        while True:
            self.local_step = 0
            if episode >= self.eval_episode:
                break
            episode += 1
            eval_reward = 0
            obs_sim = self.env_sim.reset()
            obs_real = self.env_real.reset()

            done = False

            while not done:
                self.local_step += 1
                time_start = time.time()
                input_real = self.algorithm.eval_action(obs_real)
                net_time =  time.time() - time_start

                if self.discrete == False:
                    action_real = denomalize(input_real, self.max_action, self.min_action)
                else:
                    action_real = input_real

                time_start = time.time()
                next_obs, reward, done, _ = self.car.step(action_real)
                # next_obs_real, reward_real, done_real, _ = self.env_real.step(action_real)
                sim_time = time.time() - time_start

                # if self.local_step > 10:
                #     data = np.hstack((net_time, sim_time))
                #     plot(data, label=['net', 'sim'])

                if self.render == True:
                    self.env_real.render()

                eval_reward += reward
                obs_real = next_obs

                if self.local_step == 8000:
                    print(episode, "th pos :", obs_real[0:7])
                    alive_cnt += 1

            reward_list.append(eval_reward)
        save_data(self.args.path, "error_data")
        print(
            "Eval  | Average Reward {:.2f}, Max reward: {:.2f}, Min reward: {:.2f}, Stddev reward: {:.2f}, alive rate : {:.2f}".format(
                sum(reward_list) / len(reward_list), max(reward_list), min(reward_list), np.std(reward_list),
                100 * (alive_cnt / self.eval_episode)))
        self.test_env.close()


