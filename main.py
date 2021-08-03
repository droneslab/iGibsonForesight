"""
Train on any task/scene combo from the iGibson Challenge 2021 using the Locobot robot.

Default hyperparameters are used for all algorithms, other than a buffer size of 500
 (as opposed to the default of 1_000_000) for DDPG, SAC and TD3 due to memory limitations.

Reward functions and action spaces are specified by or according to iGibson.
"""

import os
import logging
import numpy as np
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

        self.observation_space = spaces.Box(low=0, high=255, shape=(180, 320, 3), dtype=np.uint8)
        self.action_space = spaces.Box(shape=(2,), low=-1.0, high=1.0, dtype=np.float32)

        self.config_filename = os.path.join(EVAL_CONFIG_FOLDER, f'{self._robot_name}_{task}_{environment_name}.yaml')
        self.experiment_name = f'{self._robot_name}_{task}_{environment_name}'  # tensorboard logdir

        self.algorithm = algorithm
        self.training_steps = training_steps
        self.save_freq = save_freq

        self.callback_verbose = callback_verbose

        self.train(rendering_mode)

    def train(self, mode):

        print(f'TRAINING ON CONFIG FILE: {self.config_filename}')

        env = get_wrapped_env(self.config_filename, self.observation_space, self.action_space, mode=mode)

        ckpt_callback = CheckpointCallback(save_freq=self.save_freq, save_path='./models/', name_prefix=f'{self.experiment_name}_{self.algorithm.__name__}')
        rt_callback = RolloutTimeCallback(verbose=self.callback_verbose)
        hs_callback = HardwareStatsCallback(verbose=0)

        tb_log = f'./logs/{self.experiment_name}'
        if self.algorithm in [DDPG, TD3, SAC]:
            model = self.algorithm('CnnPolicy', env, buffer_size=500, verbose=1, tensorboard_log=tb_log)
        else:  # [A2C, PPO]
            model = self.algorithm('CnnPolicy', env, verbose=1, tensorboard_log=tb_log)

        model.learn(total_timesteps=self.training_steps, callback=[ckpt_callback, rt_callback, hs_callback])

        env.close()


if __name__ == '__main__':

    x = LocobotEnvironmentTrainer(algorithm=DDPG,
                                  environment_name='Beechwood_1_int',
                                  task='social_nav',
                                  rendering_mode='headless',
                                  callback_verbose=1)
