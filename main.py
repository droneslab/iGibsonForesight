"""
Train on any task/scene combo from the iGibson Challenge 2021 using the Locobot robot.

Default hyperparameters are used for all algorithms, other than a buffer size of 500
 (as opposed to the default of 1_000_000) for DDPG, SAC and TD3 due to memory limitations.

Reward functions and action spaces are specified by or according to iGibson.
"""

import os
import argparse
import logging
from datetime import datetime
import numpy as np
import torch
from gym import spaces
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.callbacks import CheckpointCallback

from util import get_wrapped_env, RolloutTimeCallback, HardwareStatsCallback

EVAL_CONFIG_FOLDER = './eval_configs'

# 8 environments included in iGibson Challenge 2021
IGC2021_ENVS = (
    'Rs_int',
    'Beechwood_1_int',
    'Benevolence_0_int',
    'Ihlen_0_int',
    'Ihlen_1_int',
    'Merom_0_int',
    'Pomaria_0_int',
    'Wainscott_1_int'
)

IGC2021_TASKS = ('interactive_nav', 'social_nav')

ALGORITHMS = (A2C, DDPG, PPO, SAC, TD3)  # Others are not valid due to Locobot's continuous action space


class LocobotEnvironmentTrainer:

    def __init__(self, algorithm, environment_name='Rs_int', task='interactive_nav',
                 training_steps=100_000, save_freq=20_000,
                 igibson_logging_level=logging.ERROR, rendering_mode='headless', callback_verbose=1):

        assert algorithm in ALGORITHMS, 'ERROR: Invalid algorithm.'
        assert environment_name in IGC2021_ENVS, 'ERROR: Invalid environment name.'
        assert task in IGC2021_TASKS, 'ERROR:  Invalid task selected.'

        logging.getLogger().setLevel(igibson_logging_level)

        self._robot_name = 'locobot'

        self.start_time = datetime.now().strftime('%m.%d.%Y-%H.%M.%S')

        self.observation_space = spaces.Box(low=0, high=255, shape=(180, 320, 3), dtype=np.uint8)
        self.action_space = spaces.Box(shape=(2,), low=-1.0, high=1.0, dtype=np.float32)

        self.config_filename = os.path.join(EVAL_CONFIG_FOLDER, f'{self._robot_name}_{task}_{environment_name}.yaml')
        self.experiment_name = f'{self._robot_name}_{task}_{environment_name}_{self.start_time}'  # tensorboard logdir

        self.algorithm = algorithm
        self.training_steps = training_steps
        self.save_freq = save_freq

        self.callback_verbose = callback_verbose

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.train(rendering_mode)

    def train(self, mode):

        print(f'TRAINING ON CONFIG FILE: {self.config_filename}')

        env = get_wrapped_env(self.config_filename, self.observation_space, self.action_space, mode=mode)

        ckpt_callback = CheckpointCallback(save_freq=self.save_freq, save_path='./models/', name_prefix=f'{self.experiment_name}_{self.algorithm.__name__}')
        rt_callback = RolloutTimeCallback(verbose=self.callback_verbose)
        hs_callback = HardwareStatsCallback(verbose=0)

        tb_log = f'./logs/{self.experiment_name}'
        if self.algorithm in [DDPG, TD3, SAC]:
            model = self.algorithm('CnnPolicy', env, buffer_size=500, verbose=0, tensorboard_log=tb_log, device=self.device)
        else:  # [A2C, PPO]
            model = self.algorithm('CnnPolicy', env, verbose=0, tensorboard_log=tb_log, device=self.device)

        model.learn(total_timesteps=self.training_steps, callback=[ckpt_callback, rt_callback, hs_callback])

        env.close()


def command_line():
    parser = argparse.ArgumentParser(description='iGibson Foresight training script (social_nav task only).')
    parser.add_argument('-a', '--algo', type=str, required=True, help='DDPG, PPO or A2C')
    parser.add_argument('-e', '--env', type=str, required=True, help='Rs_int, Beechwood_1_int, or Wainscott_1_int')
    args = parser.parse_args()

    if args.algo == 'DDPG':
        algo = DDPG
    if args.algo == 'PPO':
        algo = PPO
    if args.algo == 'A2C':
        algo = A2C

    x = LocobotEnvironmentTrainer(algorithm=algo,
                                  ernvironment_name=args.env,
                                  task='interactive_nav',
                                  rendering_mode='gui',
                                  callback_verbose=1)


def debug():
    algo = PPO
    env = 'Benevolence_0_int'
    x = LocobotEnvironmentTrainer(algorithm=algo,
                                  environment_name=env,
                                  task='interactive_nav',
                                  rendering_mode='gui',
                                  callback_verbose=1)


if __name__ == '__main__':
    debug()

