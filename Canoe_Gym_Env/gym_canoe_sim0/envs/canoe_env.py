import sys
from gym_canoe_sim0.solver_c import *
from gym_canoe_sim0.controller import *
from random import uniform
import gym
from gym import spaces
import gym_canoe_sim0.render_opengl as gl

# main code
@dataclass
class CanoeEnvParams:
    dt: float64 = 0.05
    diff: float64 = 0.0
    visc: float64 = 0.005
    force: float64 = 5.0
    source: float64 = 100.0
    window_res: uint16 = 64
    window_size: uint16 = window_res + 2
    reward_angle_factor: float64 = 0.001  # very small for now
    canoe_shape: tuple = (1., 5.)
    stay_alive_reward: float64 = 1.1 * (window_size**2 + reward_angle_factor * np.pi)
    
    canoe_init_pose: Pose = Pose()
    target_pose: Pose = Pose()
    canoe_init_vel: Pose = Pose()

    def __init__(self, canoe_init_pose=Pose(12., 13., 0.), target_pose=Pose(52., 56., 0), canoe_init_vel=Pose(0, 0, 0)):
        self.canoe_init_pose = canoe_init_pose
        self.target_pose = target_pose
        self.canoe_init_vel = canoe_init_vel


class ConversionFactor:
    def __init__(self, old_range, new_range):
        self.old_min, self.old_max = old_range
        self.new_min, self.new_max = new_range
        self.old_range = self.old_max - self.old_min
        self.new_range = self.new_max - self.new_min

    def convert(self, num):
        return (((num - self.old_min) * self.new_range) / self.old_range) + self.new_min


class CanoeEnv(gym.Env):
    metadata = {'render.modes': ['state', 'tf', 'opengl']}
    def __init__(self, params=CanoeEnvParams(), use_opengl=False):
        super(CanoeEnv, self).__init__()
        self.use_opengl = use_opengl
        self.params = params
        self.reset()

        # code for constant velocity source
        # if len(sys.argv) > 1 and sys.argv[1] == '1':
        #     self.controller = OpenLoopController(boat)
        # else:
        #     self.controller = EmptyController(boat)
        # boat = Boat((3, 9), 300, Pose(40, 40, 7*pi/6))

        # precompute some factors
        obs_range = (-5, 5)
        self.pose_point_factor = ConversionFactor((0, self.params.window_size), obs_range)
        self.pose_theta_factor = ConversionFactor((-pi, pi), obs_range)
        self.linear_vel_factor = ConversionFactor((-self.boat.linear_vel_saturation, self.boat.linear_vel_saturation), obs_range)
        self.angular_vel_factor = ConversionFactor((-self.boat.angular_vel_saturation, self.boat.angular_vel_saturation), obs_range)
        self.paddle_theta_factor = [ConversionFactor(paddle.THETA_MAX, obs_range) for paddle in self.boat.all_paddle_list]

        action_range = (-1, 1)
        self.paddle_action_factor = [ConversionFactor(action_range, (-paddle.ANGULAR_VEL_MAX, paddle.ANGULAR_VEL_MAX)) for paddle in self.boat.all_paddle_list]
        
        self.observation_space = spaces.Box(low=obs_range[0], high=obs_range[1], shape=(6+len(self.paddle_theta_factor),), dtype=float64)
        self.action_space = spaces.Box(low=action_range[0], high=action_range[1], shape=(len(self.paddle_action_factor),), dtype=float64)

        if use_opengl:
            self.gl_window = gl.open_glut_window()
        return

    def reset(self):
        self.vel = VelField(self.params.window_size, self.params.window_size)
        self.vel_new_source = copy(self.vel)

        self.tf = TransformTree()
        self.boat = Boat(self.tf, self.params.canoe_shape, 300/4, self.params.canoe_init_pose, vel=self.params.canoe_init_vel)  # n was 300
        self.done = False
        self.time = 0
        return

    def _setNextAction(self, action):
        assert len(action) == len(self.boat.all_paddle_list)
        for i in range(len(self.boat.all_paddle_list)):
            self.boat.all_paddle_list[i].setAngularVel(self.paddle_action_factor[i].convert(action[i]))
        return
    
    def _constructObservation(self):
        obs = np.zeros(10)

        pose = self.boat.pose
        obs[0] = self.pose_point_factor.convert(pose.point[0])
        obs[1] = self.pose_point_factor.convert(pose.point[1])
        obs[2] = self.pose_theta_factor.convert(pose.theta)

        vel = self.boat.vel
        obs[3] = self.linear_vel_factor.convert(vel.point[0])
        obs[4] = self.linear_vel_factor.convert(vel.point[1])
        obs[5] = self.linear_vel_factor.convert(vel.theta)

        for i in range(len(self.boat.all_paddle_list)):
            obs[6+i] = self.paddle_theta_factor[i].convert(self.boat.all_paddle_list[i].theta)

        return obs
    
    def _calculateReward(self):
        reward = self.params.stay_alive_reward -np.linalg.norm(self.boat.pose.point - self.params.target_pose.point) - self.params.reward_angle_factor * abs(angleWrap(self.boat.pose.theta-self.params.target_pose.theta))
        return reward

    def step(self, action):
        if self.done:
            print("WARNING: simulation is done")
        else:
            # add an artificial velocity source
            # for i in range(0, 10):
            #     vel_new_source.v[30+i, 20] = 0.5
            #     vel_new_source.u[30+i, 20] = 0.5

            vel_step(self.params.window_res, self.vel, self.vel_new_source, self.params.visc, self.params.dt, self.boat)

            self._setNextAction(action)
            self.done = not self.boat.stepForward(self.vel, self.params.dt)
        
        self.time += 1
        observation = self._constructObservation()
        reward = self._calculateReward()
        return (observation, reward, self.done, {})

    
    def render(self, mode='state'):
        if mode == 'state':
            print("Force: ", self.boat.getWrenches(self.vel))
            print("Vel:", self.boat.vel.point, self.boat.vel.theta)
            print("Pose:", self.boat.pose.point, self.boat.pose.theta)
            print("Done:", self.done)
        elif mode == 'tf':
            print(self.tf.renderTree())
        elif mode == 'opengl':
            assert self.use_opengl
            gl.display_func(self.gl_window, self.vel, self.boat)


if __name__ == "__main__":
    canoe_env_params = CanoeEnvParams(canoe_init_pose=Pose(12., 13., 0.), target_pose=Pose(52., 56., 0))
    canoe_env = CanoeEnv(canoe_env_params, use_opengl=True)
    done = False
    i = 0
    while not done:
        action = canoe_env.action_space.sample()
        (obs, reward, done, info) = canoe_env.step(action)
        canoe_env.render('opengl')
        print(reward)
        if i > 500:
            canoe_env.reset()
            canoe_env.render('tf')
            i = 0
        i += 1
        canoe_env.render()
