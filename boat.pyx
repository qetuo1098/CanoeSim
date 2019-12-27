from flyingcircus.extra import angles_in_ellipse
from matplotlib import patches
from tf import *
from paddle import *

class Boat:
    def __init__(self, tf, shape, discretization, pose=Pose(), vel=Pose()):
        if shape[1] < shape[0]:
            raise ValueError("Expect ellipse with dimensions a <= b")

        self.tf = tf
        self.SHAPE = shape  # (a = major_axis, b = minor_axis)
        self.pose = copy(pose)  # pose of canoe. Note: DO NOT DIRECTLY EDIT self.pose
        self.vel = copy(vel)  # linear and angular vel

        # rotation matrix
        self.C = angleToC(pose.theta)
        self.patch = patches.Ellipse(pose.point, self.SHAPE[0], self.SHAPE[1], pose.theta)  # polygon to outline the boat

        # angles from 0 to 2pi, discretized st. the arc lengths between consecutive angles are the same
        self.discretized_angles = angles_in_ellipse(discretization, self.SHAPE[0], self.SHAPE[1])

        # circumference points in the boat frame
        self.default_circumference_points = np.vstack((self.SHAPE[0]*cos(self.discretized_angles),
                                                       self.SHAPE[1]*sin(self.discretized_angles))).T

        # find normals to the ellipse surface using finite difference approximation
        # find vector of difference between previous and next point, take the (CW) normal, then normalize
        rolled_points = np.roll(self.default_circumference_points.T, -2, axis=1)
        diff = np.roll(rolled_points - self.default_circumference_points.T, 1, axis=1)
        for i in range(diff.shape[1]):
            diff[:, i] = orthogonal(diff[:, i])
        angle_vectors = normalize(diff, axis=0)

        self.canoe_frame = self.tf.constructFrame(FrameID.BOAT, self.tf.root, self.pose.point, self.pose.theta, np.copy(self.default_circumference_points.T), angle_vectors)

        # radius (distance from centre to point) of each circumference point (regardless of frame)
        self.circumference_points_radii = sqrt(square(self.default_circumference_points[:, 0]) +
                                               square(self.default_circumference_points[:, 1]))

        # force scaling from velocity to move the canoe. ToDo: make this scale common with the other forces (mouse)
        self.force_scale_v = 3/100
        self.force_scale_w = 3/10000

        # paddle
        self.paddle = Paddle(pose=Pose(0, self.SHAPE[1], pi/2), theta_max=(-pi/2, pi/2), length=15.0, discretization=150, frame_id=FrameID.PADDLE_L1, parent_frame=self.canoe_frame, tf=tf)
        self.paddle2 = Paddle(pose=Pose(0, -self.SHAPE[1], -pi/2), theta_max=(-pi/2, pi/2), length=15.0, discretization=150, frame_id=FrameID.PADDLE_R1, parent_frame=self.canoe_frame, tf=tf)
        self.paddle_list = [self.paddle, self.paddle2]
        self.tf.renderTree()

    def setPose(self, pose):
        self.pose = copy(pose).wrapAngle()
        self.C = angleToC(pose.theta)
        self.patch = patches.Ellipse(pose.point, self.SHAPE[0], self.SHAPE[1], pose.theta)
        return

    def moveByPose(self, delta_pose):
        # move canoe by pose in a frame wrt to its current pose
        self.setPose(self.pose + delta_pose)
        self.tf.changeFrame(self.canoe_frame, self.tf.root, self.pose.point, self.pose.theta)
        return

    def propelBoat(self, dt):
        # propel canoe by its current velocity by time dt
        self.moveByPose(self.vel * dt)
        return

    def getWrenches(self, vel_field):
        total_force = np.array([0.0, 0.0])
        total_torque = 0.0
        boat_force, boat_torque = calculateWrenches(vel_field, self.tf.getTransformedPoses(self.canoe_frame, self.tf.root)[0].T, inverseH(self.canoe_frame.H))
        # print(boat_force, boat_torque)
        total_force += boat_force
        total_torque += boat_torque

        for paddle in self.paddle_list:
            paddle_force, paddle_torque = calculateWrenches(vel_field, self.tf.getTransformedPoses(paddle.frame, self.tf.root)[0].T, inverseH(self.canoe_frame.H))
            paddle_vel_canoe_frame, paddle_points_canoe_frame = self.tf.getTransformedVelocities(paddle.frame, self.canoe_frame)
            opposing_force, opposing_torque = calculateOpposingWrenches(paddle_vel_canoe_frame.T, paddle_points_canoe_frame.T)
            total_force += (paddle_force + opposing_force)
            total_torque += (paddle_torque + opposing_torque)
        return total_force, total_torque

    def stepForward(self, vel_field, dt):
        vel_damper = 0.6  # todo move it somewhere else, or replace this with a general method of damping
        # update paddle location
        for paddle in self.paddle_list:
            paddle.twistPaddle(dt)

        # first update velocity from force
        total_forces, total_torque = self.getWrenches(vel_field)
        self.vel.point += total_forces * self.force_scale_v
        self.vel.theta += total_torque * self.force_scale_w

        # then damp with friction
        self.vel.point *= vel_damper
        self.vel.theta *= vel_damper

        # new: update tf
        self.canoe_frame.v = self.vel.point.reshape(2, 1)
        self.canoe_frame.w = self.vel.theta

        # then propel canoe forward
        self.propelBoat(dt)
        return


def calculateWrenches(vel_field, points_world, H_canoe):  # todo: refactor main_angle
    """
    Gets the current force and torque on the canoe in a given velocity field.
    Force: sums up the velocities at all points on the circumference (surface integral of velocities)
    Torque: sums up the cross product of the radii of all points on the circumference with velocities at these
    points (r x f)
    :param vel_field: VelField
    :return: (forces, torque): forces = (2,) np array of forces, torque wrt canoe frame
    """
    total_forces = np.zeros(2)
    total_torque = 0
    points_wrt_canoe = (H_canoe @ np.vstack((points_world.T, np.ones(points_world.shape[0]))))[:-1, :].T
    C_canoe = H_canoe[:2, :2]
    distances_wrt_canoe = np.linalg.norm(points_wrt_canoe, axis=1)
    angles_wrt_canoe = np.arctan2(points_wrt_canoe[:, 1], points_wrt_canoe[:, 0])
    for i in range(points_world.shape[0]):
        index = tuple(points_world[i])
        v_world = np.array([bilinear_interp(vel_field.u, index[0], index[1]), bilinear_interp(vel_field.v, index[0], index[1])])
        v_canoe = C_canoe @ v_world
        # torque = radius x force = |r||f|sin(angle from r to f)
        torque = distances_wrt_canoe[i] * np.linalg.norm(v_canoe) * \
                 sin(np.arctan2(v_canoe[1], v_canoe[0]) - angles_wrt_canoe[i])
        # print("torque angles:", np.arctan2(v_canoe[1], v_canoe[0]), v_canoe*1000)  # just get it from tf
        total_forces += v_canoe
        total_torque += torque
        # print("torque:", torque)

    return total_forces, total_torque

def calculateOpposingWrenches(paddle_vel_canoe_frame, points_canoe_frame):  # todo: refactor
    """
    Gets the current force and torque on the canoe in a given velocity field.
    Force: sums up the velocities at all points on the circumference (surface integral of velocities)
    Torque: sums up the cross product of the radii of all points on the circumference with velocities at these
    points (r x f)
    :param vel_field: VelField
    :return: (forces, torque): forces = (2,) np array of forces, torque = float64 total torque
    """
    paddle_opposing_wrench_scalar = 4E-2
    total_forces = np.zeros(2)
    total_torque = 0
    for i in range(points_canoe_frame.shape[0]):
        point = points_canoe_frame[i]
        forcex, forcey = paddle_vel_canoe_frame[i]
        distance = np.linalg.norm(point)
        point_angle = np.arctan2(point[1], point[0])
        # torque = radius x force = |r||f|sin(angle from r to f)
        torque = distance * np.linalg.norm(np.array([forcex, forcey])) * sin(np.arctan2(forcey, forcex) - point_angle)
        total_forces[0] += forcex
        total_forces[1] += forcey
        total_torque += torque

    return (-paddle_opposing_wrench_scalar*total_forces, -paddle_opposing_wrench_scalar*total_torque)

def propelVelField(vel_field, boat):
    """
    Change velocity field based on canoe's movement
    :param vel_field: VelField
    :param boat: Boat
    :return: None
    """
    scaling = 5E-5
    # canoe vel
    boat_circumference_points = boat.tf.getTransformedPoses(boat.canoe_frame, boat.tf.root)[0]
    for i in range(len(boat_circumference_points)):
        point = boat_circumference_points[i]
        point_vel = boat.vel.point + boat.vel.theta * boat.circumference_points_radii[i]  # linear + angular vel
        vel_field.u[int(point[0]), int(point[1])] += scaling * point_vel[0]
        vel_field.v[int(point[0]), int(point[1])] += scaling * point_vel[1]

    # paddle vel
    for paddle in boat.paddle_list:
        paddle_vel_world_frame, paddle_points_world_frame = boat.tf.getTransformedVelocities(paddle.frame, boat.tf.root)
        paddle_vel_world_frame = paddle_points_world_frame.T
        paddle_points_world_frame = paddle_points_world_frame.T
        for i in range(len(paddle_points_world_frame)):
            point = paddle_points_world_frame[i]
            point_vel = paddle_vel_world_frame[i]
            vel_field.u[int(point[0]), int(point[1])] += scaling * point_vel[0]
            vel_field.v[int(point[0]), int(point[1])] += scaling * point_vel[1]
