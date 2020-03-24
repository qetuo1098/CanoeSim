import gym
import gym_canoe_sim0

from stable_baselines import A2C
from stable_baselines import PPO2
from stable_baselines.common import make_vec_env
# from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.policies import MlpPolicy
# from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv


# Create environment
# env = gym.make('canoe_sim-v0')
if __name__ == "__main__":
    env = gym.make('canoe_sim-v0')
    model = PPO2.load("ppo2_canoe")
    # model = A2C.load("a2c_canoe")

    # Enjoy trained agent
    obs = env.reset()
    done = False
    env.openOpenGL()
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        print(reward)
        env.render('opengl')
