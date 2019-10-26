import numpy as np
import enum
from dataclasses import dataclass, field
from copy import copy
from numpy import sin, cos, square, sqrt, isclose
from numpy import float64
from misc_methods import *

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

    def __init__(self, x=0.0, y=0.0, theta=0.0):
        self.point = np.array([x, y], dtype=float64)
        self.theta = float64(theta)

    def __copy__(self):
        return Pose(self.point[0], self.point[1], self.theta)

    def __iadd__(self, other):
        C = angleToC(-self.theta)
        print(self.point, (C @ other.point.reshape((2,1))).ravel())
        self.point += (C @ other.point.reshape((2,1))).ravel()
        self.theta += other.theta
        self.theta = angleWrap(self.theta)
        return self

    def __add__(self, other):
        new_pose = copy(self)
        new_pose += other
        return new_pose


@dataclass
class VelField:
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
        new_vel_flow = VelField(self.sizex, self.sizey)
        new_vel_flow.u = np.copy(self.u)
        new_vel_flow.v = np.copy(self.v)
        return new_vel_flow

