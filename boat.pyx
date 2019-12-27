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
        self.vel = copy(vel)  # vel of canoe wrt world, written wrt to canoe frame

        # rotation matrix
        self.C = angleToC(pose.theta)
        self.patch = patches.Ellipse(pose.point, self.SHAPE[0], self.SHAPE[1], pose.theta)  # polygon to outline the boat

        # angles from 0 to 2pi, discretized st. the arc lengths between consecutive angles are the same
        self.discretized_angles = angles_in_ellipse(discretization, self.SHAPE[0], self.SHAPE[1])

        # circumference points in the boat frame
        self.default_circumference_points = np.vstack((self.SHAPE[0]*cos(self.discretized_angles),
                                                       self.SHAPE[1]*sin(self.discretized_angles))).T

        # find normals to the ellipse surface using finite difference approximation
        # find vector of difference between previous and next point, take the (CW) normal (direction doesn't matter), then normalize
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
        self.force_scale_v = 6/100
        self.force_scale_w = 6/10000

        # paddle
        self.handleL = EndPaddle(pose=Pose(self.SHAPE[0], 0, 0), theta_max=(-pi/2, pi/2), length=5.0, discretization=25, frame_id=FrameID.PADDLE_L1, parent_frame=self.canoe_frame, tf=tf)
        self.handleR = EndPaddle(pose=Pose(-self.SHAPE[0], 0, pi), theta_max=(-pi/2, pi/2), length=5.0, discretization=25, frame_id=FrameID.PADDLE_R1, parent_frame=self.canoe_frame, tf=tf)
        self.paddleL = MiddlePaddle(pose=Pose(5, 0, pi/2), theta_max=(-pi/2, pi/2), length=10.0, discretization=50, frame_id=FrameID.PADDLE_L2, parent_frame=self.handleL.frame, tf=tf)
        self.paddleR = MiddlePaddle(pose=Pose(5, 0, pi/2), theta_max=(-pi/2, pi/2), length=10.0, discretization=50, frame_id=FrameID.PADDLE_R2, parent_frame=self.handleR.frame, tf=tf)

        self.effective_paddle_list = [self.paddleL, self.paddleR]  # "in the water", forces affect the canoe
        self.all_paddle_list = [self.handleL, self.handleR, self.paddleL, self.paddleR]  # both in and out of water
        self.all_effective_frames = [self.canoe_frame, self.paddleL.frame, self.paddleR.frame]
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
        boat_points_world_frame, boat_normals_world_frame = self.tf.getTransformedPoses(self.canoe_frame, self.tf.root)
        boat_vel_world_frame, boat_points_world_frame = self.tf.getTransformedVelocities(self.canoe_frame, self.tf.root)
        boat_points_canoe_frame, _ = self.tf.getTransformedPoses(self.canoe_frame, self.canoe_frame)
        boat_force, boat_torque = calculateWrenches(vel_field, boat_vel_world_frame.T, boat_points_world_frame.T, boat_normals_world_frame.T, boat_points_canoe_frame.T, inverseH(self.canoe_frame.H))
        total_force += boat_force
        total_torque += boat_torque

        for paddle in self.effective_paddle_list:
            paddle_points_world_frame, paddle_normals_world_frame = self.tf.getTransformedPoses(paddle.frame, self.tf.root)
            paddle_vel_world_frame, paddle_points_world_frame = self.tf.getTransformedVelocities(paddle.frame, self.tf.root)
            paddle_points_canoe_frame, _ = self.tf.getTransformedPoses(paddle.frame, self.canoe_frame)
            paddle_force, paddle_torque = calculateWrenches(vel_field, paddle_vel_world_frame.T, paddle_points_world_frame.T,
                                                         paddle_normals_world_frame.T, paddle_points_canoe_frame.T,
                                                         inverseH(self.canoe_frame.H))
            total_force += paddle_force
            total_torque += paddle_torque
        return total_force, total_torque


    def stepForward(self, vel_field, dt):
        # update paddle location
        for paddle in self.all_paddle_list:
            paddle.twistPaddle(dt)

        # first update velocity from force
        total_forces, total_torque = self.getWrenches(vel_field)
        self.vel.point += total_forces * self.force_scale_v
        self.vel.theta += total_torque * self.force_scale_w

        # update tf
        self.canoe_frame.v = self.canoe_frame.H[:2, :2] @ self.vel.point.reshape(2, 1)
        self.canoe_frame.w = self.vel.theta

        # then propel canoe forward
        self.propelBoat(dt)
        return


def calculateWrenches(vel_field, vel_world_frame, points_world, normals_world_frame, points_canoe_frame, H_canoe):
    paddle_opposing_wrench_scalar = 8E-2
    total_forces = np.zeros(2)
    total_torque = 0
    C_canoe = H_canoe[:2, :2]
    points_canoe_frame_written_wrt_world = (C_canoe.T @ points_canoe_frame.T).T
    distances_wrt_canoe = np.linalg.norm(points_canoe_frame_written_wrt_world, axis=1)
    angles_wrt_canoe = np.arctan2(points_canoe_frame_written_wrt_world[:, 1], points_canoe_frame_written_wrt_world[:, 0])
    # print(vel_world_frame)
    # effect = False
    for i in range(points_world.shape[0]):
        index = tuple(points_world[i])
        if (index[0] <= 3 or index[0] >= vel_field.sizex-3 or index[1] <= 3 or index[1] >= vel_field.sizey-3):
            v_world = -vel_world_frame[i]*1
            print(index, vel_field.sizex, vel_field.sizey, v_world)
            # effect = True
        else:
            v_world = np.array([bilinear_interp(vel_field.u, index[0], index[1]), bilinear_interp(vel_field.v, index[0], index[1])]) - vel_world_frame[i] * paddle_opposing_wrench_scalar
        f_world = convertVelToForce(v_world, normals_world_frame[i])
        # torque = radius x force = |r||f|sin(angle from r to f)
        torque = distances_wrt_canoe[i] * np.linalg.norm(f_world) * \
                 sin(np.arctan2(f_world[1], f_world[0]) - angles_wrt_canoe[i])
        total_forces += f_world
        total_torque += torque
    total_forces = C_canoe @ total_forces
    return total_forces, total_torque


def propelVelField(vel_field, boat):
    """
    Change velocity field based on canoe's movement
    :param vel_field: VelField
    :param boat: Boat
    :return: None
    """
    scaling = 5E-4  # if using v^2, use 5E-5
    for object_frame in boat.all_effective_frames:
        object_vel_world_frame, object_points_world_frame = boat.tf.getTransformedVelocities(object_frame, boat.tf.root)
        object_points_world_frame, object_normals_world_frame = boat.tf.getTransformedPoses(object_frame, boat.tf.root)
        object_vel_world_frame = object_vel_world_frame.T
        object_points_world_frame = object_points_world_frame.T
        object_normals_world_frame = object_normals_world_frame.T
        for i in range(len(object_points_world_frame)):
            point = object_points_world_frame[i]
            point_vel = object_vel_world_frame[i]
            force = convertVelToForce(point_vel - np.array([vel_field.u[int(point[0]), int(point[1])], vel_field.v[int(point[0]), int(point[1])]]), object_normals_world_frame[i])
            vel_field.u[int(point[0]), int(point[1])] += scaling * force[0]
            vel_field.v[int(point[0]), int(point[1])] += scaling * force[1]
