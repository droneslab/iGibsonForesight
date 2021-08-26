from stable_baselines3 import A2C

from main import LocobotEnvironmentTrainer

if __name__ == '__main__':

    x = LocobotEnvironmentTrainer(algorithm=A2C,
                                  environment_name='Rs_int',
                                  task='interactive_nav',
                                  rendering_mode='headless',
                                  callback_verbose=1)
