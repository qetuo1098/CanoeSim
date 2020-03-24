import gym
import gym_canoe_sim0
from gym_canoe_sim0.includes import *

# Create environment
# env = gym.make('canoe_sim-v0')
if __name__ == "__main__":
    env = gym.make('canoe_sim-v0')
    obs = env.reset()
    done = False
    baseline_cont = OpenLoopController()
    env.openOpenGL()
    while not done:
        action = baseline_cont.outputCommand()
        obs, reward, done, info = env.step(action)
        print(reward)
        env.render('opengl')
