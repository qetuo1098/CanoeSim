import gym
import gym_canoe_sim0
import tensorflow as tf

from stable_baselines import A2C
# from stable_baselines import PPO2
from stable_baselines.common import make_vec_env
# from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.policies import MlpPolicy
# from stable_baselines.common.policies import MlpLnLstmPolicy
# from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv


# Create environment
# env = gym.make('canoe_sim-v0')
if __name__ == "__main__":
    env = make_vec_env('canoe_sim-v0', n_envs=32, vec_env_cls=SubprocVecEnv)

    # Instantiate the agent
    # policy_kwargs = dict(act_fun=tf.nn.relu, net_arch=[32, 32, 32])
    model = A2C(MlpPolicy, env, verbose=1, gamma=0.95, learning_rate=0.0025) #, policy_kwargs=policy_kwargs)
    # Train the agent
    model.learn(total_timesteps=int(54000*20))
    model.save("a2c_canoe")
