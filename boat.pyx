from types_common import *
from flyingcircus.extra import angles_in_ellipse
from matplotlib import patches


class Boat:
    def __init__(self, shape, discretization, pose=Pose(), vel=Twist()):
        if shape[1] < shape[0]:
            raise ValueError("Expect ellipse with dimensions b <= a")

        self.SHAPE = shape  # (a = major_axis, b = minor_axis)
        self.pose = pose
        self.vel = vel  # linear and angular vel

        # rotation matrix
        self.C = angleToC(pose.theta)

        self.patch = patches.Ellipse(pose.point, self.SHAPE[0], self.SHAPE[1], pose.theta)  # polygon to outline the boat

        # circumference points
        discretized_angles = angles_in_ellipse(discretization, self.SHAPE[0], self.SHAPE[1])
        self.default_circumference_points = np.vstack((self.SHAPE[0]*cos(discretized_angles),
                                                       self.SHAPE[1]*sin(discretized_angles))).T
        self.circumference_points = np.empty(self.default_circumference_points.shape)
        self.updateCircumferencePoints()
        self.circumference_points_radii = sqrt(square(self.circumference_points[:, 0]) +
                                               square(self.circumference_points[:, 1]))

    def setPose(self, pose):
        if not isclose(pose.theta, self.pose.theta):
            self.C = angleToC(pose.theta)
        if not isclose(pose.theta, self.pose.theta) or not isclose(pose.point, self.pose.point).all():
            self.updateCircumferencePoints()
        self.pose = pose
        return

    def updateCircumferencePoints(self):
        # get the points around the circumference of the ellipse
        # transforms self.discretized_circumference to world frame
        for i in range(self.circumference_points.shape[0]):
            self.circumference_points[i] = (self.C @ self.default_circumference_points[i].reshape((2, 1)) +
                                            self.pose.point.reshape((2, 1))).T
        return

    def getForces(self, vel_field):
        """
        Sums up the velocities at all points. When the points outlines a rigid body, it becomes a surface integral of
        velocities
        :param points: Nx2, N=number of points to integrate over
        :param vel_field: VelField
        :return: forces: (2,) np array of forces
        """
        total_forces = np.zeros(2)
        total_torque = 0
        for i in range(self.circumference_points.shape[0]):
            index = tuple(self.circumference_points[i])
            forcex = bilinear_interp(vel_field.u, index[0], index[1])
            forcey = bilinear_interp(vel_field.v, index[0], index[1])
            # torque = radius x force = |r||f|sin(angle from r to f)
            torque = self.circumference_points_radii[i] * np.linalg.norm(np.array([forcex, forcey])) * \
                     sin(np.arctan2(forcey, forcex) + self.pose.theta)
            total_forces[0] += forcex
            total_forces[1] += forcey
            total_torque += torque

        return total_forces, total_torque

    def movePose(self, pose):
        self.setPose(self.pose + pose)
