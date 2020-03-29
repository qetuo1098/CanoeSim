import gym
import gym_canoe_sim0
import tensorflow as tf
import os

from stable_baselines import PPO2
from stable_baselines.common import make_vec_env
# from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.policies import MlpPolicy
# from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize


# Create environment
# env = gym.make('canoe_sim-v0')
if __name__ == "__main__":
    env = make_vec_env('canoe_sim-v0', n_envs=32, vec_env_cls=SubprocVecEnv)
    env = VecNormalize(env)
    
    files = [f.split('.')[0] for f in os.listdir('.') if os.path.isfile(f) and f[:10] == "ppo2_canoe"]
    largest = max([int(f_num[10:]) for f_num in files])
    print("Continuing training for " + "ppo2_canoe"+str(largest))
    model = PPO2.load("ppo2_canoe"+str(largest), tensorboard_log="./ppo2_canoe_tensorboard/")
    model.set_env(env)

    # Train the agent
    model.learn(total_timesteps=int(54000*20))
    model.save("ppo2_canoe"+str(largest+1))
    print("Model saved as " + "ppo2_canoe"+str(largest+1))