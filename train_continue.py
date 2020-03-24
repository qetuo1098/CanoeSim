import gym
import gym_canoe_sim0

from stable_baselines import PPO2
from stable_baselines.common import make_vec_env
# from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.policies import MlpPolicy
# from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv


# Create environment
# env = gym.make('canoe_sim-v0')
if __name__ == "__main__":
    env = make_vec_env('canoe_sim-v0', n_envs=32, vec_env_cls=SubprocVecEnv)
    model = PPO2.load("ppo2_canoe")
    model.set_env(env)

    # Train the agent
    model.learn(total_timesteps=int(54000*5))
    model.save("ppo2_canoe")