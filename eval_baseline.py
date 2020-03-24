import gym
import gym_canoe_sim0
from gym_canoe_sim0.includes import *

# Create environment
# env = gym.make('canoe_sim-v0')
if __name__ == "__main__":
    env = gym.make('canoe_sim-v0')
    obs = env.reset()
    done = False
    total_reward = 0
    gamma = 0.995
    curr_gamma = 1.
    baseline_cont = OpenLoopController()
    env.openOpenGL()
    while not done:
        action = baseline_cont.outputCommand()
        obs, reward, done, info = env.step(action)
        total_reward += reward * curr_gamma
        curr_gamma *= gamma
        print(reward, total_reward, curr_gamma)
        env.render('opengl')
