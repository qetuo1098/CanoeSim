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

        self.force_saturation = 300
        self.torque_saturation = 3000

        self.linear_vel_saturation = 25
        self.angular_vel_saturation = 4.5

        # self.patch = patches.Ellipse(pose.point, self.SHAPE[0], self.SHAPE[1], pose.theta)  # polygon to outline the boat

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

        # force scaling from velocity to move the canoe. ToDo: make this scale common with the other forces (mouse)
        self.force_scale_v = 8*12/100
        self.force_scale_w = 8*36/10000

        # paddles
        # handles are out of the water, paddles are in the water
        self.handleL = EndPaddle(pose=Pose(0, self.SHAPE[1]-1, 0), theta_max=(-pi/4, pi+pi/4), angular_vel_max=6, length=1.25, discretization=12, frame_id=FrameID.PADDLE_L1, parent_frame=self.canoe_frame, tf=tf)
        self.handleR = EndPaddle(pose=Pose(0, -self.SHAPE[1]+1, pi), theta_max=(-pi/4, pi+pi/4), angular_vel_max=6, length=1.25, discretization=12, frame_id=FrameID.PADDLE_R1, parent_frame=self.canoe_frame, tf=tf)
        self.paddleL = MiddlePaddle(pose=Pose(1.25, 0, pi/2), theta_max=(-pi/2, pi/2), angular_vel_max=6, length=5, discretization=25, frame_id=FrameID.PADDLE_L2, parent_frame=self.handleL.frame, tf=tf)
        self.paddleR = MiddlePaddle(pose=Pose(1.25, 0, pi/2), theta_max=(-pi/2, pi/2), angular_vel_max=6, length=5, discretization=25, frame_id=FrameID.PADDLE_R2, parent_frame=self.handleR.frame, tf=tf)

        self.effective_paddle_set = {self.paddleL, self.paddleR}  # "in the water", forces affect the canoe
        self.all_paddle_list = [self.handleL, self.handleR, self.paddleL, self.paddleR]  # both in and out of water
        self.all_effective_frames = {self.canoe_frame, self.paddleL.frame, self.paddleR.frame}  # all objects that affect the forces
        self.all_frames = {self.canoe_frame} | {obj.frame for obj in self.all_paddle_list}

        # self.tf.renderTree()
        self.out_of_bounds_debug = None


    def setPose(self, pose):
        self.pose = copy(pose).wrapAngle()
        # self.patch = patches.Ellipse(pose.point, self.SHAPE[0], self.SHAPE[1], pose.theta)
        return

    def moveByPose(self, delta_pose):
        """
        Move canoe by delta_pose
        :param delta_pose: difference of pose to be added, written wrt canoe frame
        """
        self.setPose(self.pose + delta_pose)  # add delta_pose to pose directly, since pose addition adds the delta_pose in the frame of the first pose
        self.tf.changeFrame(self.canoe_frame, self.tf.root, self.pose.point, self.pose.theta)
        return

    def propelBoat(self, dt):
        # propel canoe by its current velocity by time dt
        self.moveByPose(self.vel * dt)
        return

    def getWrenches(self, vel_field):
        """
        Get the total force and torque on both the canoe and paddles due to velocity field and the movement of the canoe
        itself, by using calculateWrenches.
        :param vel_field: Velocity field
        :return: total_force (1x2), total torque (scalar)
        """
        total_force = np.array([0.0, 0.0])
        total_torque = 0.0
        boat_points_world_frame, boat_normals_world_frame = self.tf.getTransformedPoses(self.canoe_frame, self.tf.root)
        boat_vel_world_frame, boat_points_world_frame = self.tf.getTransformedVelocities(self.canoe_frame, self.tf.root)
        boat_points_canoe_frame, _ = self.tf.getTransformedPoses(self.canoe_frame, self.canoe_frame)
        boat_force, boat_torque = self.calculateWrenches(vel_field, boat_vel_world_frame.T, boat_points_world_frame.T, boat_normals_world_frame.T, boat_points_canoe_frame.T, inverseH(self.canoe_frame.H))
        total_force += boat_force
        total_torque += boat_torque

        for paddle in self.effective_paddle_set:
            paddle_points_world_frame, paddle_normals_world_frame = self.tf.getTransformedPoses(paddle.frame, self.tf.root)
            paddle_vel_world_frame, paddle_points_world_frame = self.tf.getTransformedVelocities(paddle.frame, self.tf.root)
            paddle_points_canoe_frame, _ = self.tf.getTransformedPoses(paddle.frame, self.canoe_frame)
            paddle_force, paddle_torque = self.calculateWrenches(vel_field, paddle_vel_world_frame.T, paddle_points_world_frame.T,
                                                         paddle_normals_world_frame.T, paddle_points_canoe_frame.T,
                                                         inverseH(self.canoe_frame.H))
            total_force += paddle_force
            total_torque += paddle_torque

        total_force, total_torque = saturateWrench(total_force, total_torque, self.force_saturation, self.torque_saturation)
        return total_force, total_torque


    def stepForward(self, vel_field, dt):
        # steps forward in the canoe simulation. Called by demo.py

        # update paddle location
        for paddle in self.all_paddle_list:
            paddle.twistPaddle(dt)

        # update velocity from force
        total_forces, total_torque = self.getWrenches(vel_field)
        # can add the forces to self.vel directly, as both are written wrt canoe frame
        self.vel.point += total_forces * self.force_scale_v
        self.vel.theta += total_torque * self.force_scale_w

        # saturate velocity
        self.vel = saturateVel(self.vel, self.linear_vel_saturation, self.angular_vel_saturation)

        # update tf. For linear vel, need to transform the force into world frame before adding it to tf's v
        self.canoe_frame.v = self.canoe_frame.H[:2, :2] @ self.vel.point.reshape(2, 1)
        self.canoe_frame.w = self.vel.theta  # w doesn't matter as all frames have the same w direction

        # then propel canoe forward
        self.propelBoat(dt)

        in_bounds = self.checkInBounds(vel_field)
        return in_bounds


    def calculateWrenches(self, vel_field, vel_world_frame, points_world, normals_world_frame, points_canoe_frame, H_canoe):
        """
        Finds the total forces and torques exerted onto the object due to the velocity field and the object's movement.
        For each point, dv = (velocity of point wrt world frame written wrt world frame) - velocity field at the point in
        world frame), f = dv projected onto the point's normal, is now force of point in world frame written wrt world frame
        (see convertVelToforce() in misc_methods.pyx)
        torque = r(distance from point wrt to canoe, written wrt world frame) x f = |r||f|sin(angle from r to f)
        total_forces, total_torque is by summing over the above over all object points
        :param vel_field: velocity field
        :param vel_world_frame: Nx2. velocity of points of interest wrt world frame, written wrt world frame
        :param points_world: Nx2. positions of the points of interest wrt world frame
        :param normals_world_frame: Nx2. normals of the points of interest wrt world frame
        :param points_canoe_frame: Nx2. positions of the points of interest wrt canoe frame
        :param H_canoe: homogeneous transformation from world to canoe (H @ p_world = p_canoe)
        :return: (total_forces, total_torque). total_forces: 1x2, total_torque: scalar
        """
        paddle_opposing_wrench_scalar = 1E-2
        total_forces = np.zeros(2)
        total_torque = 0
        C_canoe = H_canoe[:2, :2]
        points_canoe_frame_written_wrt_world = (C_canoe.T @ points_canoe_frame.T).T
        distances_wrt_canoe = np.linalg.norm(points_canoe_frame_written_wrt_world, axis=1)
        angles_wrt_canoe = np.arctan2(points_canoe_frame_written_wrt_world[:, 1], points_canoe_frame_written_wrt_world[:, 0])
        # print(vel_world_frame)
        for i in range(points_world.shape[0]):
            index = tuple(points_world[i])
            v_world = np.array([bilinear_interp(vel_field.u, index[0], index[1]), bilinear_interp(vel_field.v, index[0], index[1])]) - vel_world_frame[i] * paddle_opposing_wrench_scalar
            f_world = convertVelToForce(v_world, normals_world_frame[i])
            # print("v_world:", np.array([bilinear_interp(vel_field.u, index[0], index[1]), bilinear_interp(vel_field.v, index[0], index[1])]), vel_world_frame[i])
            # torque = radius x force = |r||f|sin(angle from r to f)
            torque = distances_wrt_canoe[i] * np.linalg.norm(f_world) * \
                     sin(np.arctan2(f_world[1], f_world[0]) - angles_wrt_canoe[i])
            total_forces += f_world
            total_torque += torque
        total_forces = C_canoe @ total_forces
        return total_forces, total_torque

    def checkInBounds(self, vel_field):
        for frame in self.all_frames:
            obj_points_world_frame, _ = self.tf.getTransformedPoses(frame, self.tf.root)
            obj_points_world_frame = obj_points_world_frame.T
            for index in obj_points_world_frame:
                if (index[0] <= 0 or index[0] >= vel_field.sizex-2 or index[1] <= 0 or index[1] >= vel_field.sizey-2):
                    out_of_bounds_debug = (frame.id, frame.H, index)
                    # print("not in bounds:",frame.id, frame.H)
                    # print(index, vel_field.sizex, vel_field.sizey)
                    return False
        return True


def propelVelField(vel_field, boat):
    """
    Add velocity field feedback based on canoe's movement. For all points within the canoe and the effective paddles,
    calculate the velocity of the object point wrt world frame written wrt world frame, subtract velocity field at the
    object's position in world frame, and transform it to force (procedure is similar to calculateWrenches() above).
    Then, add scaling*force to the velocity field to add the feedback back to the velocity field.
    :param vel_field: VelField
    :param boat: Boat
    :return: None
    """
    scaling = 5E-5  # if using v, use 5E-4; if using v^2, use 5E-5
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
