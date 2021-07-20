"""
Uses Stable Baselines 3 to train a DDPG agent on the iGibson environment specified by the
`turtlebot_point_nav.yaml` config file.

This environment is prepackaged w/ the default gibson2 installation (ie. not part of iGibson Challenge 2021).
"""

import os
import sys
from datetime import datetime
import numpy as np
import gym
from gym import spaces
from stable_baselines3 import DDPG
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

import gibson2
from gibson2.envs.igibson_env import iGibsonEnv
from gibson2.render.profiler import Profiler
import logging


class iGibsonRGBWrapper(gym.Wrapper):
    """
    By default, iGibson's observations are given as an `OrderedDict`. This wrapper
    changes this so only the RGB observation is returned.
    """
    def __init__(self, env, mode='rgb'):
        super().__init__(env)
        self.env = env
        self.mode = mode

        if mode not in ['task_obs', 'rgb', 'depth', 'scan']:
            print('invalid mode')
            sys.exit(1)

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)

        next_state = next_state[self.mode]

        return next_state.copy(), reward, done, info

    def reset(self):
        state = self.env.reset()

        state = state[self.mode]
        return state.copy()


class SB3Wrapper(gym.Env):
    """
    Required to properly define observation and action spaces for the environment.
    Currently hard-coded in for `turtlebot_point_nav.yaml`.

    Rendering currently not implemented through this wrapper.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, igibson_env):

        super(SB3Wrapper, self).__init__()

        self.env = igibson_env

        # Define action and observation space as gym.spaces objects
        self.action_space = spaces.Box(shape=(2,), low=-1.0, high=1.0, dtype=np.float32)

        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(120, 160, 3), dtype=np.uint8)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        done = bool(done)
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        return obs  # reward, done, info can't be included

    def render(self, mode='human'):
        pass

    def close(self):
        self.env.close()


class RolloutTimeCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info
    """
    def __init__(self, verbose=1):
        super(RolloutTimeCallback, self).__init__(verbose)
        self.verbose = verbose

        # A rollout is the collection of environment interaction using the current policy.
        # Since we are focusing on the rendering aspect, we care more about the rollout collection time,
        # as opposed to the learning time.
        self.rollout_start_time = None
        self.rollout_end_time = None

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        if self.verbose == 1:
            self.rollout_start_time = datetime.now()

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """

        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        if self.verbose == 1:
            self.rollout_end_time = datetime.now()

            print(f'Rollout time: {self.rollout_end_time - self.rollout_start_time} seconds')

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass


def get_wrapped_env(config_filename):

    env = iGibsonEnv(config_filename, mode='headless')
    env = iGibsonRGBWrapper(env)
    env = SB3Wrapper(env)
    env = Monitor(env)

    return env


def eval_gibson_stats(env, model, max_episode_length=500):
    """
    Runs 3 episodes for 500 timesteps each, printing FPS and time info for each step.
    """

    for j in range(3):
        obs = env.reset()
        for i in range(max_episode_length):
            # env.render()  # doesn't work w/ gibson environments

            with Profiler('Environment action step'):  # displays FPS and time per observation

                # action = env.action_space.sample()
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)

                if i >= max_episode_length:
                    done = True

                if done:
                    logging.info(
                        "Episode finished after {} timesteps".format(i + 1))
                    break
    env.close()


def main(config_filename, training_steps=100_000, save_freq=20_000):
    """
    Trains agent using DDPG on environment specified by `config_filename`.
    """

    env = get_wrapped_env(config_filename)

    checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path='./models/', name_prefix='DDPG')
    rollout_time_callback = RolloutTimeCallback(verbose=1)

    model = DDPG("CnnPolicy", env, batch_size=16, buffer_size=600, verbose=1, tensorboard_log='./logs/')
    model.learn(total_timesteps=training_steps, callback=[checkpoint_callback, rollout_time_callback])

    # eval_gibson_stats(env, model, max_episode_length=500)


if __name__ == '__main__':

    logging.getLogger().setLevel(logging.ERROR)

    config_filename = os.path.join(os.path.join(gibson2.example_config_path, 'turtlebot_point_nav.yaml'))
    main(config_filename)
