import sys
import gym
from gym_canoe_sim0.includes import *
from random import uniform

# main code
np.seterr(all='raise')
@dataclass
class CanoeEnvParams:
    dt: float64 = 0.05
    diff: float64 = 0.0
    visc: float64 = 0.005
    force: float64 = 5.0
    source: float64 = 100.0
    window_res: uint16 = 64
    window_size: uint16 = window_res + 2
    reward_distance_factor: float64 = 5
    reward_angle_factor: float64 = 80  # very small for now
    reward_reach_goal: float64 = 200.
    reach_goal_radius: float64 = 4.5
    canoe_shape: tuple = (1., 5.)
    stay_alive_reward: float64 = 0.
    # stay_alive_reward: float64 = 1.1 * (window_size**2 + reward_angle_factor * np.pi)
    out_of_bounds_penalty: float64 = 100.
    max_steps: np.uint64 = 2048
    
    canoe_init_pose_range_x: tuple = (12., 55.)
    canoe_init_pose_range_y: tuple = (12., 55.)
    canoe_init_pose_range_theta: tuple = (-pi, pi)
    target_pose: Pose = Pose()
    canoe_init_vel: Pose = Pose()

    def __init__(self, target_pose=Pose(50., 50., 0), canoe_init_vel=Pose(0, 0, 0)):
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
        self._selfReset()

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

        self.reward_range = (-100, 100)

        if use_opengl:
            self.gl_window = gl.open_glut_window()
        return
    
    def setPose(self, new_pose):
        self.boat.setPose(new_pose)  # add delta_pose to pose directly, since pose addition adds the delta_pose in the frame of the first pose
        self.tf.changeFrame(self.boat.canoe_frame, self.tf.root, new_pose.point, new_pose.theta)
    
    def _sampleInitPose(self):
        # use set pose for now
        # return Pose(12., 13., 0)
        init_pose_x = uniform(*self.params.canoe_init_pose_range_x)
        init_pose_y = uniform(*self.params.canoe_init_pose_range_y)
        init_pose_theta = 0.
        # init_pose_theta = uniform(*self.params.canoe_init_pose_range_theta)
        return Pose(init_pose_x, init_pose_y, init_pose_theta)

    def openOpenGL(self):
        assert not self.use_opengl
        self.use_opengl = True
        self.gl_window = gl.open_glut_window()

    def _selfReset(self):
        self.vel = VelField(self.params.window_size, self.params.window_size)
        self.vel_new_source = copy(self.vel)

        self.tf = TransformTree()
        init_pose = self._sampleInitPose()
        self.boat = Boat(self.tf, self.params.canoe_shape, 300/4, init_pose, vel=self.params.canoe_init_vel)  # n was 300
        self.done = False
        self.time = 0
        # initialize last_distance_deviation, last_angle_deviation by calling _calculateReward() once
        self.last_distance_deviation, self.last_angle_deviation = 0, 0
        self._calculateReward()

    def reset(self):
        # return observation at reset
        self._selfReset()
        return self._constructObservation()

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
        # print(obs[:3])
        return obs
    
    def _calculateReward(self):
        deviation_from_goal = self.params.target_pose.point - self.boat.pose.point
        distance_deviation = np.linalg.norm(deviation_from_goal)
        angle_deviation = self.boat.pose.theta+np.pi/2 - np.arctan2(deviation_from_goal[1], deviation_from_goal[0])
        angle_deviation = min(abs(angleWrap(angle_deviation)), abs(angleWrap(angle_deviation + np.pi)))  # canoe head and tail doesn't matter
        # print("angle dev:", self.boat.pose.theta+np.pi/2, np.arctan2(deviation_from_goal[1], deviation_from_goal[0]), angle_deviation)

        reward = self.params.stay_alive_reward + (self.last_distance_deviation - distance_deviation) * self.params.reward_distance_factor + (self.last_angle_deviation - angle_deviation) * self.params.reward_angle_factor
        self.last_distance_deviation = distance_deviation
        self.last_angle_deviation = angle_deviation
        return reward
    
    def _checkReachGoal(self):
        return np.linalg.norm(self.boat.pose.point - self.params.target_pose.point) < self.params.reach_goal_radius

    def step(self, action):
        # print("action", action)
        if self.done:
            print("WARNING: simulation is done")
        else:
            # add an artificial velocity source
            # for i in range(0, 10):
            #     vel_new_source.v[30+i, 20] = 0.5
            #     vel_new_source.u[30+i, 20] = 0.5

            vel_step(self.params.window_res, self.vel, self.vel_new_source, self.params.visc, self.params.dt, self.boat)

            self._setNextAction(action)
            out_of_bounds = not self.boat.stepForward(self.vel, self.params.dt)

        reward = self._calculateReward()
        
        if out_of_bounds:
            reward -= self.params.out_of_bounds_penalty
        if self._checkReachGoal():
            self.done = True
            reward += self.params.reward_reach_goal
        
        if self.time > self.params.max_steps or out_of_bounds:
            self.done = True
            
        self.time += 1
        observation = self._constructObservation()
        # print("obs", observation)

        # assert self.use_opengl
        # gl.display_func(self.gl_window, self.vel, self.boat)
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
            gl.display_func(self.gl_window, self.vel, self.boat, self.params.target_pose)


if __name__ == "__main__":
    canoe_env_params = CanoeEnvParams(target_pose=Pose(52., 56., 0))
    canoe_env = CanoeEnv(canoe_env_params, use_opengl=True)
    canoe_env.setPose(Pose(12., 12., 0))
    done = False
    i = 0
    while not done:
        action = canoe_env.action_space.sample()
        (obs, reward, done, info) = canoe_env.step(action)
        canoe_env.render('opengl')
        print(reward)
        if i > 512:
            canoe_env.reset()
            canoe_env.render('tf')
            i = 0
        i += 1
        canoe_env.render()
