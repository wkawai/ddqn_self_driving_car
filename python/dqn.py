# coding: utf-8

import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import gym
import numpy as np
from collections import deque
import copy
import matplotlib.pyplot as plt
import numpy as np
import sys
from unityagents import UnityEnvironment
import pylab as pl
from datetime import datetime
import os
import argparse
import statistics

# np.set_printoptions(threshold=np.nan, linewidth=np.nan)

# class QFunction(chainer.Chain):

#     def __init__(self, obs_size, n_actions, n_hidden_channels=50):
#         super().__init__()
#         with self.init_scope():
#             self.l0 = L.Linear(obs_size, n_hidden_channels)
#             self.l1 = L.Linear(n_hidden_channels, n_hidden_channels)
#             self.l2 = L.Linear(n_hidden_channels, n_actions)

#     def __call__(self, x, test=False):
#         """
#         Args:
#             x (ndarray or chainer.Variable): An observation
#             test (bool): a flag indicating whether it is in test mode
#         """

#         h = F.tanh(self.l0(x))
#         h = F.tanh(self.l1(h))
#         return chainerrl.action_value.DiscreteActionValue(self.l2(h))


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--test-mode', action='store_true')
    parser.add_argument('--gpu', type=int, default=-1)

    args = parser.parse_args()

    for arg in vars(args):
        print('{}={}'.format(arg, getattr(args, arg)))

    # hyper parameters
    hps = {}
    # environment  parameters
    hps['obs_size'] = 40
    hps['n_actions'] = 9
    hps['n_episodes'] = 1000
    hps['max_episode_len'] = 5000

    # DQN parameters
    hps['eps'] = 1e-2
    hps['n_hidden_channels'] = 40
    hps['n_hidden_layers'] = 2
    hps['gamma'] = 0.95
    hps['capacity'] = 1e6
    hps['start_epsilon'] = 0.9
    hps['end_epsilon'] = 0.05
    hps['decay_steps'] = 5e5
    hps['replay_start_size'] = 500
    hps['update_interval'] = 1
    hps['target_update_interval'] = 100

    # Name of the Unity environment binary to launch
    env_name = "DQNSelfDrivingCar"
    # env_name = "DQNSelfDrivingCarCity"

    # Load agent file
    # agent_files = []
    if args.test_mode == True:
        # angle limit = 25
        # agent_file = os.path.join(
            # os.getcwd(), 'results', '2017_12_19_15_57_24', 'best_agent')
        # agent_files.append(agent_file)
        # agent_files = [
        #     os.path.join(os.getcwd(), 'results',
        #                  '2017_12_20_13_44_47', 'best_agent'),
        #     os.path.join(os.getcwd(), 'results',
        #                  '2017_12_20_12_57_40', 'best_agent'),
        #     os.path.join(os.getcwd(), 'results',
        #                  '2017_12_20_12_15_30', 'best_agent'),
        #     os.path.join(os.getcwd(), 'results',
        #                  '2017_12_20_11_43_04', 'best_agent'),
        #     os.path.join(os.getcwd(), 'results',
        #                  '2017_12_20_11_07_25', 'best_agent'),
        #     os.path.join(os.getcwd(), 'results',
        #                  '2017_12_20_01_32_30', 'best_agent'),
        #     os.path.join(os.getcwd(), 'results',
        #                  '2017_12_20_00_52_40', 'best_agent'),
        #     os.path.join(os.getcwd(), 'results',
        #                  '2017_12_20_00_16_30', 'best_agent'),
        #     os.path.join(os.getcwd(), 'results',
        #                  '2017_12_19_23_45_49', 'best_agent'),
        #     os.path.join(os.getcwd(), 'results',
        #                  '2017_12_19_23_07_19', 'best_agent'),

        # ]

        # angle limit = 35
        # agent_files = [
        #     os.path.join(os.getcwd(), 'results',
        #                  '2017_12_20_13_51_30', 'best_agent'),
        #     os.path.join(os.getcwd(), 'results',
        #                  '2017_12_20_14_37_47', 'best_agent'),
        #     os.path.join(os.getcwd(), 'results',
        #                  '2017_12_20_15_18_23', 'best_agent'),
        #     os.path.join(os.getcwd(), 'results',
        #                  '2017_12_20_15_56_25', 'best_agent'),
        #     os.path.join(os.getcwd(), 'results',
        #                  '2017_12_20_16_38_20', 'best_agent'),
        #     os.path.join(os.getcwd(), 'results',
        #                  '2017_12_20_17_16_16', 'best_agent'),
        #     os.path.join(os.getcwd(), 'results',
        #                  '2017_12_20_17_55_23', 'best_agent'),
        #     os.path.join(os.getcwd(), 'results',
        #                  '2017_12_20_18_38_33', 'best_agent'),
        #     os.path.join(os.getcwd(), 'results',
        #                  '2017_12_20_19_18_28', 'best_agent'),
        #     os.path.join(os.getcwd(), 'results',
        #                  '2017_12_20_19_58_57', 'best_agent'),

        # ]

        # # reward = -50
        # agent_files = [
        #     # os.path.join(os.getcwd(), 'results',
        #                 #  '2017_12_21_10_47_34', 'best_agent'),
        #     # os.path.join(os.getcwd(), 'results',
        #                 #  '2017_12_21_12_13_28', 'best_agent'),
        #     os.path.join(os.getcwd(), 'results',
        #                  '2017_12_21_13_05_33', 'best_agent'),
        #     # os.path.join(os.getcwd(), 'results',
        #     #              '2017_12_21_13_56_00', 'best_agent'),
        #     # os.path.join(os.getcwd(), 'results',
        #     #              '2017_12_21_14_45_57', 'best_agent'),
        #     # os.path.join(os.getcwd(), 'results',
        #     #              '2017_12_21_15_41_20', 'best_agent'),
        #     # os.path.join(os.getcwd(), 'results',
        #     #              '2017_12_21_16_43_49', 'best_agent'),
        #     # os.path.join(os.getcwd(), 'results',
        #     #              '2017_12_21_17_26_26', 'best_agent'),
        # ]

        # prioritized experience replay
        agent_files = [
            os.path.join(os.getcwd(), 'results',
                         '2017_12_22_10_38_44', 'best_agent'),
            #  '2017_12_22_10_38_44', 'agent100'),
        ]
        # agent_files = [
        #     os.path.join(os.getcwd(), 'results',
        #                 #  '2018_01_05_18_41_32', 'agent10'),
        #                 #  '2018_01_05_18_41_32', 'agent50'),
        #                 #  '2018_01_05_18_41_32', 'agent1000'),
        #                  '2018_01_05_18_41_32', 'best_agent'),
        # ]


        # city
        # agent_files = [
        #     os.path.join(os.getcwd(), 'results',
        #                  '2017_12_31_09_14_04', 'best_agent'),
        # ]



    if args.test_mode == True:
        n_agent = len(agent_files)
    else:
        n_agent = 1

    # Test mode parameters
    if args.test_mode == True:
        chainer.config.train = False
        hps['start_epsilon'] = 0.0
        hps['end_epsilon'] = 0.0
        hps['decay_steps'] = 1
        hps['eps'] = 0.0

    # Agent save dir
    if args.test_mode == False:
        dir_name = os.path.join(os.getcwd(), 'results',
                                datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
        print(dir_name)
        os.mkdir(dir_name)

    # Instanciate environment
    env = UnityEnvironment(file_name=env_name)

    # Examine environment parameters
    print(str(env))

    # Set the default brain to work with
    default_brain = env.brain_names[0]
    brain = env.brains[default_brain]

    # Reset the environment
    env_info = env.reset(train_mode=not(args.test_mode))[default_brain]

    # Examine the state space for the default brain
    print("Agent state looks like: \n{}".format(env_info.states[0]))

    # Examine the observation space for the default brain
    for observation in env_info.observations:
        print("Agent observations look like:")
        if observation.shape[3] == 3:
            plt.imshow(observation[0, :, :, :])
        else:
            plt.imshow(observation[0, :, :, 0])

    # Instanciate Q-function
    # q_func = QFunction(hps['obs_size'], hps['n_actions'],
            #    n_hidden_channels=hps['n_hidden_channels'])

    # Since observations from CartPole-v0 is numpy.float64 while
    # Chainer only accepts numpy.float32 by default, specify
    # a converter as a feature extractor function phi.
    def phi(x): return x.astype(np.float32, copy=False)

    agents = []

    for _ in range(n_agent):
        q_func = chainerrl.q_function.FCStateQFunctionWithDiscreteAction(
            ndim_obs=hps['obs_size'], n_actions=hps['n_actions'],
            n_hidden_channels=hps['n_hidden_channels'],
            n_hidden_layers=hps['n_hidden_layers'], nonlinearity=F.tanh)

        # q_func = chainerrl.q_function.FCLSTMStateQFunction(
        #     ndim_obs=hps['obs_size'], n_actions=hps['n_actions'],
        #     n_hidden_channels=hps['n_hidden_channels'],
        #     n_hidden_layers=hps['n_hidden_layers'], nonlinearity=F.tanh)

        # Configurre GPU
        if args.gpu >= 0:
            q_func.to_gpu(args.gpu)

        # Use Adam to optimize q_func. eps=1e-2 is for stability.
        optimizer = chainer.optimizers.Adam(eps=hps['eps'])
        optimizer.setup(q_func)
        # optimizer.add_hook(chainer.optimizer.Lasso(0.00001))

        # Use epsilon-greedy for exploration
        # explorer = chainerrl.explorers.ConstantEpsilonGreedy(
        #    epsilon=0.3, random_action_func=lambda  :np.random.randint(9) )
        explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(
            start_epsilon=hps['start_epsilon'], end_epsilon=hps['end_epsilon'], decay_steps=hps['decay_steps'], random_action_func=lambda: np.random.randint(9))

        # DQN uses Experience Replay.
        # Specify a replay buffer and its capacity.
        # replay_buffer = chainerrl.replay_buffer.ReplayBuffer(
        # capacity=int(hps['capacity']))
        replay_buffer = chainerrl.replay_buffer.PrioritizedReplayBuffer()

        # Now create an agent that will interact with the environment.
        agent = chainerrl.agents.DoubleDQN(
            q_func, optimizer, replay_buffer, hps['gamma'], explorer,
            replay_start_size=hps['replay_start_size'],
            update_interval=hps['update_interval'],
            target_update_interval=hps['target_update_interval'], phi=phi)

        agents.append(agent)

    # preprocess_matrix = np.zeros((184, 40))
    # for i in range(4):
    #     preprocess_matrix[i, i] = 1

    # for j in range(4, 40):
    #     for i in range((j - 4) * 5 + 4, (j - 4) * 5 + 4 + 5):
    #         preprocess_matrix[i, j] = 1 / 5

    # print(preprocess_matrix)

    total_step = 0
    Rs = []
    mean_Rs = []
    mean_size = 50
    max_R = -1e5

    obs_log = []
    reward_log = []

    if args.test_mode == True:
        for agent, agent_file in zip(agents, agent_files):
            agent.load(agent_file)

    for i in range(1, hps['n_episodes'] + 1):
        env_info = env.reset(train_mode=not(args.test_mode))[default_brain]
        reward = 0
        obs = env_info.states[0]
        # obs = obs @ preprocess_matrix

        done = False
        R = 0  # return (sum of rewards)
        t = 0  # time step

        while not done:
            obs_log.append(obs)
            reward_log.append(reward)

            # choose action
            if args.test_mode == True:
                # actions = []
                action_counts = np.zeros(hps['n_actions'])
                for agent in agents:
                    action = agent.act(obs)
                    # actions.append(action)
                    action_counts[action] += 1
                action = np.random.choice(np.arange(hps['n_actions'])[
                                          action_counts == np.max(action_counts)])
                print(action_counts)
                # print(actions)

                # action = np.random.choice(actions)
            else:
                action = agents[0].act_and_train(obs, reward)

            # take action and observe reward and state(obs)
            env_info = env.step(action)[default_brain]
            obs = env_info.states[0]
            # obs = obs @ preprocess_matrix

            reward = env_info.rewards[0]
            done = env_info.local_done[0]

            total_step += 1

            R += reward
            t += 1

        if (args.test_mode == False) and (R > max_R):
            # Save an agent to the 'agent' directory
            max_R = R
            agents[0].save(os.path.join(dir_name,  'best_agent'))

        if i % 10 == 0:
            print('total_step={:6}   episode={:4}   R={:5.1f}   mean_R={:5.1f}   statistics:{}'
                  .format(total_step, i, R, mean_Rs[-1], agent.get_statistics()))

        # if i % 100 == 0 and args.test_mode == False:
            # agents[0].save(os.path.join(dir_name,  'agent{}'.format(i)))
        if i in [10, 50, 100, 500, 1000] and args.test_mode == False:
            agents[0].save(os.path.join(dir_name,  'agent{}'.format(i)))

        Rs.append(R)
        mean_Rs.append(np.mean(Rs[-mean_size:]))

        if args.test_mode == True:
            for agent in agents:
                agent.stop_episode()
        else:
            agents[0].stop_episode_and_train(obs, reward, done)

    print('Finished.')

    # Plot reward
    if args.test_mode == False:
        plt.plot(Rs, label='return')
        plt.plot(mean_Rs, label='mean return (50)')
        plt.xlabel('episode')
        plt.ylabel('return')
        plt.legend()
        plt.savefig(os.path.join(dir_name, 'Return.png'), dpi=300)

    # Close environment
    env.close()

    # obs_log = np.array(obs_log)
    # reward_log = np.array(reward_log)
    # print(obs_log.shape)
    # print(reward_log.shape)

    # np.save(os.path.join(dir_name, 'obs_log.npy'), obs_log, )
    # np.save(os.path.join(dir_name, 'reward_log.npy'), reward_log)

    # write hyperparamters to file
    if args.test_mode == False:
        with open(os.path.join(dir_name, 'hyperparameters.txt'), mode='w') as file:
            for key, val in hps.items():
                file.write('{}={}\n'.format(key, val))
        file.close()


if __name__ == '__main__':
    main()
