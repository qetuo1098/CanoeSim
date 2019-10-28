import numpy as np
import enum
from dataclasses import dataclass, field
from copy import copy
from numpy import sin, cos, square, sqrt, isclose
from numpy import float64
from misc_methods import *

pi = np.pi

# Solver objects
"""
@dataclass
class Twist:
    v: float64(dtype=float64)
    w: float64(dtype=float64)

    def __init__(self):
        self.v = self.w = 0.0

    def __iadd__(self, other):
        if not isinstance(other, Twist):
            raise ValueError("Can only add Twist to a Twist")
        self.v += other.v
        self.w = angleWrap(self.w + other.w)
        return self

    def __add__(self, other):
        new_twist = self
        new_twist += other
        return new_twist

    def __imul__(self, other):
        if isinstance(other, Twist):
            raise ValueError("Cannot multiply a Twist by a Twist")
        self.v *= other
        self.w = angleWrap(self.w * other)
        return self

    def __mul__(self, other):
        new_twist = self
        new_twist *= other
        return new_twist

    def scale(self, scale):
        v_scale, w_scale = scale
        self.v *= v_scale
        self.w =  angleWrap(self.w * w_scale)
        return self
"""

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
        # the second vector is added to the first in the frame of the first
        C = angleToC(-self.theta)
        self.point += (C @ other.point.reshape((2,1))).ravel()
        self.theta += other.theta
        self.theta = angleWrap(self.theta)
        return self

    def __add__(self, other):
        new_pose = copy(self)
        new_pose += other
        return new_pose

    def __imul__(self, other):
        if isinstance(other, Pose):
            raise ValueError("Cannot multiply a Pose by a Pose. RHS must be scalar")
        self.point *= other
        self.theta = angleWrap(self.theta * other)
        return self

    def __mul__(self, other):
        new_pose = copy(self)
        new_pose *= other
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

