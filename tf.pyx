from types_common import *
from anytree import Node, NodeMixin, RenderTree
from misc_methods import *

class FrameID(enum.Enum):
    """
    Id of frames. Each frame has a unique ID.
    Only used to manually identify frames through printing out the frame.id for debugging.
    """
    WORLD = enum.auto()
    BOAT = enum.auto()
    PADDLE_L1 = enum.auto()
    PADDLE_L2 = enum.auto()
    PADDLE_R1 = enum.auto()
    PADDLE_R2 = enum.auto()


class Frame(Node):
    """
    A frame within the transform tree. Contains transformation wrt its parent frame, v wrt its parent frame, points
    within the frame that are stationary wrt this frame, and the surface normals of these points
    id: id of frame (in class FrameID)
    H: (3x3) homogeneous matrix of this frame wrt to its parent frame, written wrt parent frame
    pose_points: (2xN) points of this frame
    pose_angle_vectors: (2xN) surface normals (in terms of unit vectors) of points in this frame
    v: (1x2) velocity of this frame wrt parent frame, written wrt parent frame
    w: (scalar) angular velocity wrt parent frame (written wrt to any frame, as the rotation is in 2D)
    """
    def __init__(self, id, parent_frame, H, pose_points, pose_angle_vectors, v, w):
        super().__init__(id, parent_frame)
        self.id = id
        self.H = H
        self.pose_points = pose_points
        self.pose_angle_vectors = pose_angle_vectors
        self.v = v.reshape((2, 1))
        self.w = w

    def changeAngle(self, angle):
        # change the transformation's angle (may be easier than changing H directly), doesn't change translation
        self.H[:2, :2] = angleToC(angle)


class TransformTree:
    def __init__(self):
        self.root = self.constructFrame(FrameID.WORLD, None, np.zeros(2), 0)  # world frame

    def constructFrame(self, id, parent_frame, t, theta, pose_points=np.ndarray((2, 0)), pose_angle_vectors=np.ndarray((2, 0)), v=np.zeros(2), w=0):
        """
        Construct a new frame in the transform tree, linked to its specified parent frame with the specified
        transformations, frame points, and velocities.
        :param id: frame id from FrameID enum class
        :param parent_frame: parent frame of the frame to be constructed
        :param t: (2x1), translation of the new frame wrt its parent frame (written wrt parent frame)
        :param theta: (scalar), rotation of the new frame wrt its parent frame (written wrt parent frame)
        :param pose_points: (2xN), points within this frame, and will not move wrt this frame
        :param pose_angle_vectors: (2xN), normal vectors of the surfaces formed by the points within this frame
        :param v: (1x2), linear velocity vector of this frame wrt its parent frame, written wrt its parent frame
        :param w: (scalar), angular velocity of this frame wrt its parent frame, written wrt any frame (just a scalar for 2D)
        :return: new_frame: the new frame constructed
        """
        H = constructH(t, theta)
        if pose_angle_vectors.size == 0:
            pose_angle_vectors = np.zeros(pose_points.shape)
            pose_angle_vectors[1, :].fill(1)  # construct angle vectors of all (0, 1), i.e. upward, for the paddles points

        new_frame = Frame(id, parent_frame, H, pose_points, pose_angle_vectors, v, w)
        return new_frame  # reference of the node for easier identification later

    def changeFrame(self, frame, frame_parent, t, theta):
        """
        Change the homogeneous transformation between frame and its parent frame, using the new transformation information
        t and theta. frame_parent is only passed in as an error checker, as it is only possible to change transformations
        between two immediate frames, and not between frames with intermediate frames in the middle
        :param frame: frame whose transformation is to be changed
        :param frame_parent: parent frame of frame
        :param t: 2x1
        :param theta: scalar
        """
        if frame_parent != frame.parent:
            raise ValueError(str(frame_parent) + " is not a parent of " + str(frame))
        frame.H = constructH(t, theta)
        return

    def getTransform(self, f1, f0):
        """
        Get the homogeneous transformation matrix of frame f1 (not the points in f1) wrt frame f0.
        Transformation is f1 wrt f0 written in f0 frame
        :param f1: frame
        :param f0: frame, must be ancestor of f1
        :return: H, 3x3 homogeneous transformation matrix
        """
        if f0 not in f1.path:
            raise ValueError(str(f0) + " is not in the path of " + str(f1))
        # calculate transform
        H = np.eye(3)
        for frame in f1.iter_path_reverse():
            if frame == f0:
                break
            H = frame.H @ H
        return H

    def getTransformedPoses(self, f1, f0):
        """
        Get positions of points and the normals of points in f1 wrt f0 written in the frame of f0
        :param f1: frame of which we wish to find the positions/normals of the points wrt f0
        :param f0: frame, of which we wish to find f1 points/normals wrt to and written wrt to. f0 must be an ancestor of f1
        :return: (points, normals). points: 2xN. normals: 2xN
        """
        H = self.getTransform(f1, f0)
        homog_points = np.vstack((f1.pose_points, np.ones(f1.pose_points.shape[1])))
        transformed_points = (H @ homog_points)[:-1, :]
        transformed_angle_vectors = H[:2, :2] @ f1.pose_angle_vectors
        return transformed_points, transformed_angle_vectors

    def getTransformedVelocities(self, f1, f0):
        """
        Get velocities and positions of points in f1 wrt f0 written in the frame of f0
        :param f1: frame of which we wish to find the velocities/positions of the points
        :param f0: frame. f0 must be an ancestor of f1
        :return: (v, points). v: velocities, 2xN. points: positions, 2xN
        """
        if f0 not in f1.path:
            raise ValueError(str(f0) + " is not in the path of " + str(f1))
        # calculate transform
        H = np.eye(3, dtype=float64)
        points = np.vstack((f1.pose_points, np.ones(f1.pose_points.shape[1])))
        v = np.zeros(f1.pose_points.shape, dtype=float64)
        for frame in f1.iter_path_reverse():
            # using the current frame to calculate the v and points of the frame wrt the parent frame
            if frame == f0:
                break
            H = frame.H @ H
            frame_C = frame.H[:2, :2]
            # derived from r02 = r01 + C01 * r12
            # r02_dot = r01_dot + C01 * r12_dot + (w01 x C01) * r12
            # here, r12 is the position of the point wrt the last frame (1), so r12_dot is always 0
            # then, the loop would start at frame 1, calculating r02_dot with v, w, H of frame 1 wrt frame 0
            # with more linked frames, just iterate as before
            # also need to calculate r12, r02, etc. (points) at each iteration by using H
            v = np.repeat(frame.v, points.shape[1], axis=1) + frame_C @ v + wcrossC(frame.w, frame_C) @ points[:-1, :]
            points = frame.H @ points
        return v, points[:-1, :]

    def renderTree(self):
        # print out the entire transform tree
        # starting from the root and going depth-first down the tree, print the frame and all variables within each frame
        print(RenderTree(self.root))
