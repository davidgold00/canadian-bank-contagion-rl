def make_ppo(env):
    from stable_baselines3 import PPO
    return PPO('MlpPolicy', env, verbose=0)
def make_dqn(env):
    from stable_baselines3 import DQN
    return DQN('MlpPolicy', env, verbose=0)
