import sys
from datetime import datetime
import gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from gibson2.envs.igibson_env import iGibsonEnv
from gibson2.render.profiler import Profiler
import logging


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


class iGibsonRGBWrapper(gym.Wrapper):
    """
    By default, iGibson's observations are given as an `OrderedDict`. This wrapper
    changes this so only the RGB observation is returned.
    """
    def __init__(self, env, mode='rgb'):

        assert mode == 'rgb', 'Currently only mode=rgb is supported.'

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

    def __init__(self, igibson_env, observation_space, action_space):

        super(SB3Wrapper, self).__init__()

        self.env = igibson_env

        self.observation_space = observation_space
        self.action_space = action_space

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


def get_wrapped_env(config_filename, observation_space, action_space, mode='headless'):

    assert mode in ['gui', 'headless', 'iggui', 'pbgui'], 'Invalid mode selected.'

    env = iGibsonEnv(config_filename, mode=mode)
    env = iGibsonRGBWrapper(env)
    env = SB3Wrapper(env, observation_space, action_space)
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
