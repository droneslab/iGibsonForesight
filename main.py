import os
import logging
import numpy as np
from gym import spaces
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.callbacks import CheckpointCallback

from util import get_wrapped_env, RolloutTimeCallback

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

ALGORITHMS = (A2C, DDPG, PPO, SAC, TD3)


class LocobotEnvironmentTrainer:

    def __init__(self, algorithm=DDPG, environment_name='Rs_int', task='interactive_nav', training_steps=100_000, save_freq=20_000,
                 igibson_logging_level=logging.ERROR):

        assert algorithm in ALGORITHMS, 'ERROR: Invalid algorithm.'
        assert environment_name in IGC2021_ENVS, 'ERROR: Invalid environment name.'

        logging.getLogger().setLevel(igibson_logging_level)

        self._robot_name = 'locobot'

        self.observation_space = spaces.Box(low=0, high=255, shape=(180, 320, 3), dtype=np.uint8)
        self.action_space = spaces.Box(shape=(2,), low=-1.0, high=1.0, dtype=np.float32)

        # TO DO: CREATE CONFIG FILES
        self.config_filename = os.path.join(EVAL_CONFIG_FOLDER, f'{self._robot_name}_{task}_{environment_name}.yaml')
        self.experiment_name = f'{self._robot_name}_{environment_name}'  # tensorboard logdir

        self.algorithm = algorithm
        self.training_steps = training_steps
        self.save_freq = save_freq

        # self.train()

    def train(self, mode='headless'):

        env = get_wrapped_env(self.config_filename, self.observation_space, self.action_space, mode=mode)

        checkpoint_callback = CheckpointCallback(save_freq=self.save_freq, save_path='./models/', name_prefix=self.experiment_name)
        rollout_time_callback = RolloutTimeCallback(verbose=1)

        model = self.algorithm('CnnPolicy', env, verbose=1, tensorboard_log=f'./logs/{self.experiment_name}')
        model.learn(total_timesteps=self.training_steps, callback=[checkpoint_callback, rollout_time_callback])



if __name__ == '__main__':

    x = LocobotEnvironmentTrainer()