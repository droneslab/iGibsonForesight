"""
Uses Stable Baselines 3 to train a DDPG agent on the iGibson environment specified by the
`turtlebot_point_nav.yaml` config file.

This environment is prepackaged w/ the default gibson2 installation (ie. not part of iGibson Challenge 2021).
"""

import os
import logging
import numpy as np

from gym import spaces
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import CheckpointCallback
import gibson2


from util import get_wrapped_env, RolloutTimeCallback


def main(config_filename, training_steps=100_000, save_freq=20_000):
    """
    Trains agent using DDPG on environment specified by `config_filename`.
    """

    experiment_name = config_filename.split('/')[-1][:-5] + f'_DDPG'

    # Example for using image as input (channel-first; channel-last also works):
    observation_space = spaces.Box(low=0, high=255, shape=(120, 160, 3), dtype=np.uint8)

    action_space = spaces.Box(shape=(2,), low=-1.0, high=1.0, dtype=np.float32)

    env = get_wrapped_env(config_filename, observation_space, action_space)

    checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path='./models/', name_prefix=experiment_name)
    rollout_time_callback = RolloutTimeCallback(verbose=1)

    model = DDPG("CnnPolicy", env, batch_size=16, buffer_size=600, verbose=1, tensorboard_log=f'./logs/{experiment_name}')
    model.learn(total_timesteps=training_steps, callback=[checkpoint_callback, rollout_time_callback])


if __name__ == '__main__':

    logging.getLogger().setLevel(logging.ERROR)

    config_filename = os.path.join(os.path.join(gibson2.example_config_path, 'turtlebot_point_nav.yaml'))
    main(config_filename)
