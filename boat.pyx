from types_common import *
from flyingcircus.extra import angles_in_ellipse
from matplotlib import patches

class Paddle:
    def __init__(self, pose, theta_max, length, discretization):
        """
        :param pose: neutral pose (theta=0) in canoe frame, or the transformation from paddle to canoe frame
        :param theta_max: min and max angle in paddle frame
        :param length: length of paddle
        :param discretization: number of points in the paddle
        """
        self.pose = pose
        self.THETA_MAX = theta_max
        self.LENGTH = length
        self.DISCRETIZATION = discretization
        self.theta = 0
        self.angular_vel = 0
        self.current_canoe_C = np.identity(2)
        self.current_canoe_point = np.zeros(2)
        self.current_canoe_vel = Pose()

        # make discretization lengths, a column vector from 0 to self.length
        self.discretization_lengths = (self.LENGTH / (self.DISCRETIZATION + 1) * np.arange(0, self.DISCRETIZATION + 2))\
            .reshape(self.DISCRETIZATION + 2, 1)
        self.points_canoe_frame = np.zeros((self.DISCRETIZATION + 2, 2))
        self.points_world_frame = np.zeros(self.points_canoe_frame.shape)
        self.point_radii = np.zeros(self.DISCRETIZATION + 2)
        self.point_angles = np.zeros(self.DISCRETIZATION + 2)
        self.vel_world_frame = np.zeros(self.points_canoe_frame.shape)

        # construct C matrix
        self.C = np.zeros((2,2))
        self.updateTransformation()
        self.twistPaddle()
        return

    def twistPaddle(self):
        scalar = 0.01  # ToDo move this somewhere
        dtheta = self.angular_vel * scalar
        # ToDo: angleWrap this. For now, min/max on a wrapped angle may cause problems
        self.theta = min(self.THETA_MAX[1], max(self.THETA_MAX[0], self.theta + dtheta))
        if close(self.theta, self.THETA_MAX[0]) or close(self.theta, self.THETA_MAX[1]):
            # saturate angular vel
            self.angular_vel = 0
        self.updateTransformation()
        self.updatePoints()
        return

    def updateTransformation(self):
        self.C = angleToC(self.pose.theta + self.theta)
        return

    def updatePoints(self):
        unit_paddle = np.array([1.0, 0.0])
        points_paddle_frame = self.discretization_lengths @ unit_paddle.reshape(1, 2)
        if points_paddle_frame.shape != self.points_canoe_frame.shape:  # extra check, can remove this later
            raise ValueError("paddle frame and canoe frame matrices do not match")
        # update points on the paddle (in canoe frame)
        for i in range(len(self.points_canoe_frame)):
            self.points_canoe_frame[i] = (self.C @ points_paddle_frame[i].reshape((2, 1)) +
                                          self.pose.point.reshape((2, 1))).T
        # print(self.points_canoe_frame[-1])
        # update points_world_frame
        for i in range(len(self.points_canoe_frame)):
            self.points_world_frame[i] = (self.current_canoe_C @ self.points_canoe_frame[i].reshape((2, 1))
                                          + self.current_canoe_point.reshape((2, 1))).T

        # find radii and angles from centre of canoe for all points
        differences = np.empty(self.points_canoe_frame.shape)
        for i in range(len(self.points_canoe_frame)):
            differences[i] = self.points_canoe_frame[i] - self.pose.point
        self.point_radii = np.linalg.norm(differences, axis=1)
        differences = differences.T
        self.point_angles = np.arctan2(differences[1], differences[0])

        canoe_vel_world_frame = (self.current_canoe_C.T @ self.current_canoe_vel.point.reshape(2, 1)).reshape(1, 2)
        for i in range(len(self.points_canoe_frame)):
            paddle_vel_canoe_rotation = self.current_canoe_vel.theta * \
                                        orthogonal(self.points_world_frame[i] - self.current_canoe_point)
            paddle_vel_paddle_movement = self.angular_vel * orthogonal(self.points_canoe_frame[i])
            self.vel_world_frame[i] = canoe_vel_world_frame + paddle_vel_canoe_rotation + paddle_vel_paddle_movement

        # print(self.vel_world_frame[-1])
        # print(self.points_world_frame[0])
        return

    def updateWorldFrameTransform(self, new_canoe_C, new_canoe_point, new_canoe_vel):
        self.current_canoe_C = new_canoe_C
        self.current_canoe_point = new_canoe_point
        self.current_canoe_vel = new_canoe_vel
        return


class Boat:
    def __init__(self, shape, discretization, pose=Pose(), vel=Pose()):
        if shape[1] < shape[0]:
            raise ValueError("Expect ellipse with dimensions a <= b")

        self.SHAPE = shape  # (a = major_axis, b = minor_axis)
        self.pose = copy(pose)  # pose of canoe. Note: DO NOT DIRECTLY EDIT self.pose
        self.vel = copy(vel)  # linear and angular vel

        # rotation matrix
        self.C = angleToC(pose.theta)

        self.patch = patches.Ellipse(pose.point, self.SHAPE[0], self.SHAPE[1], pose.theta)  # polygon to outline the boat

        # circumference points

        # angles from 0 to 2pi, discretized st. the arc lengths between consecutive angles are the same
        self.discretized_angles = angles_in_ellipse(discretization, self.SHAPE[0], self.SHAPE[1])

        # circumference points in the boat frame
        self.default_circumference_points = np.vstack((self.SHAPE[0]*cos(self.discretized_angles),
                                                       self.SHAPE[1]*sin(self.discretized_angles))).T

        # circumference points in the world frame
        self.circumference_points = np.empty(self.default_circumference_points.shape)
        self.updateCircumferencePoints()

        # radius (distance from centre to point) of each circumference point (regardless of frame)
        self.circumference_points_radii = sqrt(square(self.circumference_points[:, 0]) +
                                               square(self.circumference_points[:, 1]))

        # force scaling from velocity to move the canoe. ToDo: make this scale common with the other forces (mouse)
        self.force_scale_v = 3/100
        self.force_scale_w = 3/10000

        # paddle
        self.paddle = Paddle(pose=Pose(0, self.SHAPE[1], pi/2), theta_max=(-pi/2, pi/2), length=10.0, discretization=100)
        self.paddle2 = Paddle(pose=Pose(0, -self.SHAPE[1], -pi/2), theta_max=(-pi/2, pi/2), length=10.0, discretization=100)
        self.paddle_list = [self.paddle, self.paddle2]

    def setPose(self, pose):
        self.pose = copy(pose).wrapAngle()
        self.C = angleToC(pose.theta)
        self.updateCircumferencePoints()
        self.patch = patches.Ellipse(pose.point, self.SHAPE[0], self.SHAPE[1], pose.theta)
        for paddle in self.paddle_list:
            paddle.updateWorldFrameTransform(self.C, self.pose.point, self.vel)
        return

    def updateCircumferencePoints(self):
        # get the points around the circumference of the ellipse
        # transforms self.discretized_circumference to world frame
        for i in range(self.circumference_points.shape[0]):
            self.circumference_points[i] = (self.C @ self.default_circumference_points[i].reshape((2, 1)) +
                                            self.pose.point.reshape((2, 1))).T
        return

    def moveByPose(self, pose):
        # move canoe by pose in a frame wrt to its current pose
        self.setPose(self.pose + pose)
        return

    def propelBoat(self, dt):
        # propel canoe by its current velocity by time dt
        self.moveByPose(self.vel * dt)
        return

    def getWrenches(self, vel_field):
        total_force = total_torque = 0.0

        boat_force, boat_torque = calculateWrenches(vel_field, self.circumference_points, self.circumference_points_radii, self.discretized_angles, self.pose.theta)
        total_force += boat_force
        total_torque += boat_torque

        for paddle in self.paddle_list:
            paddle_force, paddle_torque = calculateWrenches(vel_field, paddle.points_world_frame, paddle.point_radii, paddle.point_angles, self.pose.theta)
            opposing_force, opposing_torque = calculateOpposingWrenches(paddle.vel_world_frame, paddle.points_world_frame, paddle.point_radii, paddle.point_angles, self.pose.theta)
            total_force += (paddle_force + opposing_force)
            total_torque += (paddle_torque + opposing_torque)
        return total_force, total_torque

    def stepForward(self, vel_field, dt):
        vel_damper = 0.90  # todo move it somewhere else, or replace this with a general method of damping
        # update paddle location
        for paddle in self.paddle_list:
            paddle.twistPaddle()

        # first update velocity from force
        total_forces, total_torque = self.getWrenches(vel_field)
        self.vel.point += self.C @ total_forces * self.force_scale_v  # previous bug: didn't transform force into canoe frame
        self.vel.theta += total_torque * self.force_scale_w

        # then damp with friction
        self.vel *= vel_damper

        # then propel canoe forward
        self.propelBoat(dt)
        return


def calculateWrenches(vel_field, points, distances, point_angles, main_angle=0.0):  # todo: refactor main_angle
    """
    Gets the current force and torque on the canoe in a given velocity field.
    Force: sums up the velocities at all points on the circumference (surface integral of velocities)
    Torque: sums up the cross product of the radii of all points on the circumference with velocities at these
    points (r x f)
    :param vel_field: VelField
    :return: (forces, torque): forces = (2,) np array of forces, torque = float64 total torque
    """
    total_forces = np.zeros(2)
    total_torque = 0
    for i in range(points.shape[0]):
        index = tuple(points[i])
        forcex = bilinear_interp(vel_field.u, index[0], index[1])
        forcey = bilinear_interp(vel_field.v, index[0], index[1])
        # torque = radius x force = |r||f|sin(angle from r to f)
        torque = distances[i] * np.linalg.norm(np.array([forcex, forcey])) * \
                 sin(np.arctan2(forcey, forcex) - (point_angles[i] + main_angle))
        total_forces[0] += forcex
        total_forces[1] += forcey
        total_torque += torque

    return total_forces, total_torque

def calculateOpposingWrenches(paddle_vel, points, distances, point_angles, main_angle=0.0):  # todo: refactor
    """
    Gets the current force and torque on the canoe in a given velocity field.
    Force: sums up the velocities at all points on the circumference (surface integral of velocities)
    Torque: sums up the cross product of the radii of all points on the circumference with velocities at these
    points (r x f)
    :param vel_field: VelField
    :return: (forces, torque): forces = (2,) np array of forces, torque = float64 total torque
    """
    paddle_opposing_wrench_scalar = 5E-3
    total_forces = np.zeros(2)
    total_torque = 0
    for i in range(points.shape[0]):
        forcex, forcey = paddle_vel[i]
        # torque = radius x force = |r||f|sin(angle from r to f)
        torque = distances[i] * np.linalg.norm(np.array([forcex, forcey])) * \
                 sin(np.arctan2(forcey, forcex) - (point_angles[i] + main_angle))
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
    for i in range(len(boat.circumference_points)):
        point = boat.circumference_points[i]
        point_vel = boat.vel.point + boat.vel.theta * boat.circumference_points_radii[i]  # linear + angular vel
        vel_field.u[int(point[0]), int(point[1])] += scaling * point_vel[0]
        vel_field.v[int(point[0]), int(point[1])] += scaling * point_vel[1]

    # paddle vel
    for paddle in boat.paddle_list:
        for i in range(len(paddle.points_world_frame)):
            point = paddle.points_world_frame[i]
            point_vel = paddle.vel_world_frame[i]
            vel_field.u[int(point[0]), int(point[1])] += scaling * point_vel[0]
            vel_field.v[int(point[0]), int(point[1])] += scaling * point_vel[1]
