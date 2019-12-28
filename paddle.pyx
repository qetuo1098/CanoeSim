from types_common import *
from tf import *

class Paddle:
    def __init__(self, pose, theta_max, length, discretization, frame_id, parent_frame, tf):
        """
        Paddle object. Used for both paddles (in water) and handles (out of water)
        :param pose: pose of transformation from its parent frame to the paddle frame (written wrt parent frame)
        :param theta_max: min and max angle in paddle frame
        :param length: length of paddle
        :param discretization: number of points in the paddle
        :param frame_id: id assigned to self.frame (to be created in __init__)
        :param parent_frame: frame of which the paddle is attached to
        """
        self.pose = pose
        self.THETA_MAX = theta_max
        self.LENGTH = length
        self.DISCRETIZATION = discretization
        self.theta = 0  # current angle of the paddle wrt its neutral position
        self.angular_vel = 0

        # make discretization lengths, a column vector from 0 to self.length
        self.discretization_lengths = (self.LENGTH / (self.DISCRETIZATION + 1) * np.arange(0, self.DISCRETIZATION + 2))\
            .reshape(self.DISCRETIZATION + 2, 1)

        self.tf = tf  # reference
        self.frame = tf.constructFrame(frame_id, parent_frame, pose.point, pose.theta, pose_points=np.zeros((2, self.discretization_lengths.shape[0])))  # ToDo: if possible, avoid this purely virtual construction workaround

        # construct C matrix
        self.C = np.zeros((2,2))
        self.twistPaddle(0)
        return

    def setAngularVel(self, w):
        # change the angular velocity of the paddle. Used to control paddle movements
        self.angular_vel = w
        self.frame.w = w

    def twistPaddle(self, dt):
        # simulate paddle by timestep dt with its current angular velocity
        dtheta = self.angular_vel * dt
        # ToDo: angleWrap this. For now, min/max on a wrapped angle may cause problems
        self.theta = min(self.THETA_MAX[1], max(self.THETA_MAX[0], self.theta + dtheta))
        self.frame.changeAngle(self.pose.theta + self.theta)
        if close(self.theta, self.THETA_MAX[0]) or close(self.theta, self.THETA_MAX[1]):
            # saturate theta if it reaches theta_max
            self.setAngularVel(0)
        return


class EndPaddle(Paddle):
    # paddle attached to its parent at one end of the paddle
    # paddle points go from (0, 0) to (length, 0) in the paddle frame
    def __init__(self, pose, theta_max, length, discretization, frame_id, parent_frame, tf):
        super(EndPaddle, self).__init__(pose, theta_max, length, discretization, frame_id, parent_frame, tf)
        unit_paddle = np.array([1.0, 0.0])
        self.points_paddle_frame = (self.discretization_lengths @ unit_paddle.reshape(1, 2)).T
        self.frame.pose_points = self.points_paddle_frame


class MiddlePaddle(Paddle):
    # paddle attached to its parent at the middle of the paddle
    # paddle points go from (-length/2, 0) to (length/2, 0) in the paddle frame
    def __init__(self, pose, theta_max, length, discretization, frame_id, parent_frame, tf):
        super(MiddlePaddle, self).__init__(pose, theta_max, length, discretization, frame_id, parent_frame, tf)
        unit_paddle = np.array([1.0, 0.0])
        self.points_paddle_frame = (self.discretization_lengths @ unit_paddle.reshape(1, 2)).T

        # shift the entire paddle within its frame by half of its length such that the attachment is at the middle instead of the end
        self.points_paddle_frame[0] -= self.LENGTH/2

        self.frame.pose_points = self.points_paddle_frame
