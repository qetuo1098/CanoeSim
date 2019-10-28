from types_common import *
from flyingcircus.extra import angles_in_ellipse
from matplotlib import patches


class Boat:
    def __init__(self, shape, discretization, pose=Pose(), vel=Pose()):
        if shape[1] < shape[0]:
            raise ValueError("Expect ellipse with dimensions b <= a")

        self.SHAPE = shape  # (a = major_axis, b = minor_axis)
        self.pose = copy(pose)  # pose of canoe. Note: DO NOT DIRECTLY EDIT self.pose
        self.vel = copy(vel)  # linear and angular vel

        # rotation matrix
        self.C = angleToC(pose.theta)

        self.patch = patches.Ellipse(pose.point, self.SHAPE[0], self.SHAPE[1], pose.theta)  # polygon to outline the boat

        # circumference points

        # angles from 0 to 2pi, discretized st. the arc lengths between consecutive angles are the same
        discretized_angles = angles_in_ellipse(discretization, self.SHAPE[0], self.SHAPE[1])

        # circumference points in the boat frame
        self.default_circumference_points = np.vstack((self.SHAPE[0]*cos(discretized_angles),
                                                       self.SHAPE[1]*sin(discretized_angles))).T

        # circumference points in the world frame
        self.circumference_points = np.empty(self.default_circumference_points.shape)
        self.updateCircumferencePoints()

        # radius (distance from centre to point) of each circumference point (regardless of frame)
        self.circumference_points_radii = sqrt(square(self.circumference_points[:, 0]) +
                                               square(self.circumference_points[:, 1]))

        # force scaling from velocity to move the canoe. ToDo: make this scale common with the other forces (mouse)
        self.force_scale_v = 1/100
        self.force_scale_w = 1/10000

    def setPose(self, pose):
        self.pose = copy(pose).wrapAngle()
        # if not isclose(pose.theta, self.pose.theta):
        self.C = angleToC(pose.theta)
        # if not isclose(pose.theta, self.pose.theta) or not isclose(pose.point, self.pose.point).all():
        self.updateCircumferencePoints()
        self.patch = patches.Ellipse(pose.point, self.SHAPE[0], self.SHAPE[1], pose.theta)
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
        Gets the current force and torque on the canoe in a given velocity field.
        Force: sums up the velocities at all points on the circumference (surface integral of velocities)
        Torque: sums up the cross product of the radii of all points on the circumference with velocities at these
        points (r x f)
        :param vel_field: VelField
        :return: (forces, torque): forces = (2,) np array of forces, torque = float64 total torque
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

    def moveByPose(self, pose):
        # move canoe by pose in a frame wrt to its current pose
        self.setPose(self.pose + pose)
        return

    def propelBoat(self, dt):
        # propel canoe by its current velocity by time dt
        self.moveByPose(self.vel * dt)
        return

    def stepForward(self, vel_field, dt):
        vel_damper = 0.95

        # first update velocity from force
        total_forces, total_torque = self.getForces(vel_field)
        self.vel.point += -total_forces * self.force_scale_v  # ToDo: find out why force has to be negative
        self.vel.theta += -total_torque * self.force_scale_w  # ToDo: find out why torque has to be negative

        # then damp with friction
        self.vel *= vel_damper

        # then propel canoe forward
        self.propelBoat(dt)
        return


def propelVelField(vel_field, boat):
    """
    Change velocity field based on canoe's movement
    :param vel_field: VelField
    :param boat: Boat
    :return: None
    """
    scaling = 1E-5
    for i in range(len(boat.circumference_points)):
        point = boat.circumference_points[i]
        point_vel = boat.vel.point + boat.vel.theta * boat.circumference_points_radii[i]  # linear + angular vel
        vel_field.u[int(point[0]), int(point[1])] += scaling * point_vel[0]
        vel_field.v[int(point[0]), int(point[1])] += scaling * point_vel[1]
