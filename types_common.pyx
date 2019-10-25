import numpy as np
import enum
from dataclasses import dataclass, field
from copy import copy
from matplotlib import patches
import scipy.special as sc


# Solver objects
@dataclass
class Twist:
    v: float
    w: float

    def __init__(self):
        self.v = self.w = 0.0


@dataclass
class Pose:
    x: float
    y: float
    theta: float

    def __init__(self, x=0, y=0, theta=0):
        self.x = x
        self.y = y
        self.theta = theta

    def point(self):
        return [self.x, self.y]


@dataclass
class VelFlow:
    u: np.array
    v: np.array
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
    def __init__(self, shape, pose=Pose(), vel=Twist()):
        self.SHAPE = shape  # (a, b)
        self.pose = pose
        self.vel = vel  # linear and angular vel

        self.patch = patches.Ellipse(pose.point(), shape[0], shape[1], pose.theta)  # polygon to outline the boat

