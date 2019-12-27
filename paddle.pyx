from types_common import *
from tf import *

class Paddle:
    def __init__(self, pose, theta_max, length, discretization, frame_id, parent_frame, tf):
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

        # make discretization lengths, a column vector from 0 to self.length
        self.discretization_lengths = (self.LENGTH / (self.DISCRETIZATION + 1) * np.arange(0, self.DISCRETIZATION + 2))\
            .reshape(self.DISCRETIZATION + 2, 1)

        self.tf = tf  # reference
        self.frame = tf.constructFrame(frame_id, parent_frame, pose.point, pose.theta, pose_points=np.zeros((2, self.discretization_lengths.shape[0])))  # ToDo: if possible, avoid this purely virtual construction workaround

        self.point_radii = np.zeros(self.DISCRETIZATION + 2)
        self.point_angles = np.zeros(self.DISCRETIZATION + 2)

        # construct C matrix
        self.C = np.zeros((2,2))
        self.twistPaddle(0)
        return

    def setAngularVel(self, w):
        self.angular_vel = w
        self.frame.w = w

    def twistPaddle(self, dt):
        dtheta = self.angular_vel * dt
        # ToDo: angleWrap this. For now, min/max on a wrapped angle may cause problems
        self.theta = min(self.THETA_MAX[1], max(self.THETA_MAX[0], self.theta + dtheta))
        self.frame.changeAngle(self.pose.theta + self.theta)
        if close(self.theta, self.THETA_MAX[0]) or close(self.theta, self.THETA_MAX[1]):
            # saturate theta
            self.setAngularVel(0)
        return


class EndPaddle(Paddle):
    def __init__(self, pose, theta_max, length, discretization, frame_id, parent_frame, tf):
        super(EndPaddle, self).__init__(pose, theta_max, length, discretization, frame_id, parent_frame, tf)
        unit_paddle = np.array([1.0, 0.0])
        self.points_paddle_frame = (self.discretization_lengths @ unit_paddle.reshape(1, 2)).T
        self.frame.pose_points = self.points_paddle_frame


class MiddlePaddle(Paddle):
    def __init__(self, pose, theta_max, length, discretization, frame_id, parent_frame, tf):
        super(MiddlePaddle, self).__init__(pose, theta_max, length, discretization, frame_id, parent_frame, tf)
        unit_paddle = np.array([1.0, 0.0])
        self.points_paddle_frame = (self.discretization_lengths @ unit_paddle.reshape(1, 2)).T
        self.points_paddle_frame[0] -= self.LENGTH/2
        self.frame.pose_points = self.points_paddle_frame
