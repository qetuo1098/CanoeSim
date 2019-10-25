import numpy as np
import enum
from dataclasses import dataclass, field
from copy import copy
from matplotlib import patches
from flyingcircus.extra import angles_in_ellipse
from numpy import sin, cos, isclose
from numpy import float64

pi = np.pi

# Solver objects
@dataclass
class Twist:
    v: float64(dtype=float64)
    w: float64(dtype=float64)

    def __init__(self):
        self.v = self.w = 0.0


@dataclass
class Pose:
    point: np.array(dtype=float64)
    theta: float64

    def __init__(self, x=0, y=0, theta=0):
        self.point = np.array([x, y])
        self.theta = theta


@dataclass
class VelFlow:
    u: np.array(dtype=float64)
    v: np.array(dtype=float64)
    sizex: int
    sizey: int

    def __init__(self, sizex, sizey):
        self.u = np.zeros((sizex, sizey))
        self.v = np.zeros((sizex, sizey))
        self.sizex = sizex
        self.sizey = sizey

    def __copy__(self):
        new_vel_flow = VelFlow(self.sizex, self.sizey)
        new_vel_flow.u = np.copy(self.u)
        new_vel_flow.v = np.copy(self.v)
        return new_vel_flow


class Boat:
    def __init__(self, shape, discretization, pose=Pose(), vel=Twist()):
        if shape[0] < shape[1]:
            raise ValueError("Expect ellipse with dimensions a >= b")

        self.SHAPE = shape  # (a = major_axis, b = minor_axis)
        self.pose = pose
        self.vel = vel  # linear and angular vel

        self.patch = patches.Ellipse(pose.point, self.SHAPE[0], self.SHAPE[1], pose.theta)  # polygon to outline the boat

        discretized_angles = angles_in_ellipse(discretization, self.SHAPE[1], self.SHAPE[0])
        self.discretized_circumference = np.vstack((self.SHAPE[0]*np.cos(discretized_angles),
                                                    self.SHAPE[1]*np.sin(discretized_angles))).T
        self.C = np.zeros((2, 2), dtype=float64)  # rotation matrix
        self.updateC(pose.theta)

    def updateC(self, angle):
        cosx, sinx = cos(angle), sin(angle)
        self.C = np.array([[cosx, -sinx], [sinx, cosx]])
        return

    def setPose(self, pose):
        self.pose = pose
        if not isclose(pose.theta, self.pose.theta):
            self.updateC(pose.theta)
        return

    def getBoundaryPoints(self):
        # get the points around the circumference of the ellipse
        # transforms self.discretized_circumference to world frame
        points = np.empty(self.discretized_circumference.shape)
        for i in range(self.discretized_circumference.shape[0]):
            points[i] = (self.C @ self.discretized_circumference[i].reshape((2, 1)) + self.pose.point.reshape((2, 1))).T
        return points
