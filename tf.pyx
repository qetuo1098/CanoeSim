from types_common import *
from anytree import Node, NodeMixin, RenderTree
from misc_methods import *

class FrameID(enum.Enum):
    WORLD = enum.auto()
    BOAT = enum.auto()
    PADDLE_L1 = enum.auto()
    PADDLE_L2 = enum.auto()
    PADDLE_R1 = enum.auto()
    PADDLE_R2 = enum.auto()


class Frame(Node):
    """
    id: id of frame (in class FrameID)
    H: homogeneous matrix
    pose_points: 2xN numpy array of points of this frame
    pose_angle_vectors: 2xN numpy array of angles (in terms of unit vectors) of points in this frame
    v: velocity of this frame wrt parent frame, written wrt parent frame
    w: angular velocity wrt parent frame (written wrt to any frame, as the rotation is in 2D)
    """
    # H: np.array(dtype=float64)
    # pose_points: np.array(dtype=float64)  # only used if we need the pose/vel of this specific frame. Not used in vel propagation.
    # pose_angle_vectors: np.array(dtype=float64)  # only used if we need the pose/vel of this specific frame. Not used in vel propagation.
    # v: np.array(dtype=float64)
    # w: float64

    def __init__(self, id, parent_frame, H, pose_points, pose_angle_vectors, v, w):
        super().__init__(id, parent_frame)
        self.id = id
        self.H = H
        self.pose_points = pose_points
        self.pose_angle_vectors = pose_angle_vectors
        self.v = v.reshape((2, 1))
        self.w = w

    def changeAngle(self, angle):
        self.H[:2, :2] = angleToC(angle)


class TransformTree:
    def __init__(self):
        self.root = self.constructFrame(FrameID.WORLD, None, np.zeros(2), 0)

    def constructFrame(self, id, parent_frame, t, theta, pose_points=np.ndarray((2, 0)), pose_angle_vectors=np.ndarray((2, 0)), v=np.zeros(2), w=0):
        H = constructH(t, theta)
        if pose_angle_vectors.size == 0:
            pose_angle_vectors = np.zeros(pose_points.shape)
            pose_angle_vectors[1, :].fill(1)  # construct angle vectors of all (0, 1), i.e. upward, for the paddles points

        new_frame = Frame(id, parent_frame, H, pose_points, pose_angle_vectors, v, w)
        return new_frame  # reference of the node for easier identification later

    def changeFrame(self, frame, frame_parent, t, theta):
        if frame_parent != frame.parent:
            raise ValueError(str(frame_parent) + " is not a parent of " + str(frame))
        frame.H = constructH(t, theta)

    def getTransform(self, f1, f0):
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
        H = self.getTransform(f1, f0)
        homog_points = np.vstack((f1.pose_points, np.ones(f1.pose_points.shape[1])))
        transformed_points = (H @ homog_points)[:-1, :]
        transformed_angle_vectors = H[:2, :2] @ f1.pose_angle_vectors
        return transformed_points, transformed_angle_vectors

    def getTransformedVelocities(self, f1, f0):
        if f0 not in f1.path:
            raise ValueError(str(f0) + " is not in the path of " + str(f1))
        # calculate transform
        H = np.eye(3, dtype=float64)
        points = np.vstack((f1.pose_points, np.ones(f1.pose_points.shape[1])))
        v = np.zeros(f1.pose_points.shape, dtype=float64)
        for frame in f1.iter_path_reverse():
            # print("v:",v)
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
            # print(frame.id, frame.v)
            points = frame.H @ points
        # print(v, points[:-1, :])
        return v, points[:-1, :]

    def renderTree(self):
        print(RenderTree(self.root))
