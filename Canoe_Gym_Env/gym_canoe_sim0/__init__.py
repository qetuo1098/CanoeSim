from gym.envs.registration import register

register(
    id='canoe_sim-v0',
    entry_point='gym_canoe_sim0.envs:CanoeEnv',
)
