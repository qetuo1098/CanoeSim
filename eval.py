import gym
import gym_canoe_sim0
import sys

from stable_baselines import A2C
from stable_baselines import PPO2
from stable_baselines.common import make_vec_env
# from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.policies import MlpPolicy
# from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv, VecVideoRecorder


# Create environment
# env = gym.make('canoe_sim-v0')
if __name__ == "__main__":
    video_folder = 'logs/videos/'
    video_length = 100
    env_id = 'canoe_sim-v0'
    env = gym.make(env_id)
    # env = DummyVecEnv([lambda: gym.make(env_id)])
    # env = VecVideoRecorder(env, video_folder,
    #                    record_video_trigger=lambda x: x == 0, video_length=video_length,
    #                    name_prefix="random-agent-{}".format(env_id))
    
    if len(sys.argv) > 1:
        model = PPO2.load(sys.argv[1])
    else:
        model = PPO2.load("ppo2_canoe")
    # model = A2C.load("a2c_canoe")

    # Enjoy trained agent
    obs = env.reset()
    done = False
    total_reward = 0
    gamma = 0.995
    curr_gamma = 1.
    env.openOpenGL()
    counter = 0
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward * curr_gamma
        curr_gamma *= gamma
        print(counter, reward, total_reward, curr_gamma)
        env.render('opengl')
        counter+= 1
