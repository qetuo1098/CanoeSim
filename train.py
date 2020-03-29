import gym
import gym_canoe_sim0
import tensorflow as tf

# from stable_baselines import A2C
from stable_baselines import PPO2
from stable_baselines.common import make_vec_env
# from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.policies import MlpPolicy
# from stable_baselines.common.policies import MlpLnLstmPolicy
# from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

def linear_schedule(initial_value):
    """
    Linear learning rate schedule.

    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress):
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress: (float)
        :return: (float)
        """
        return progress * initial_value

    return func


# Create environment
# env = gym.make('canoe_sim-v0')
if __name__ == "__main__":
    env = make_vec_env('canoe_sim-v0', n_envs=32, vec_env_cls=SubprocVecEnv)
    env = VecNormalize(env)
    # Instantiate the agent
    policy_kwargs = dict(act_fun=tf.nn.relu, net_arch=[64, 256, 64])
    learning_rate = 5e-4
    cliprange = 0.2
    
    model = PPO2(MlpPolicy, env, verbose=1, gamma=0.99, n_steps=512, learning_rate=linear_schedule(learning_rate), cliprange=linear_schedule(cliprange), lam=0.95, nminibatches=32, ent_coef=0.001, tensorboard_log="./ppo2_canoe_tensorboard/", policy_kwargs=policy_kwargs)
    # Train the agent
    model.learn(total_timesteps=int(54000*20))
    model.save("ppo2_canoe21")
