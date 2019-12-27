import numpy as np
from sklearn.preprocessing import normalize

def bilinear_interp(d0, x, y):
    # bilinearly interpolate value at field d0[x,y], with x,y being floats
    x0, x1 = int(x), int(x) + 1
    y0, y1 = int(y), int(y) + 1
    s0, s1 = x1 - x, x - x0
    t0, t1 = y1 - y, y - y0
    b = (s0 * (t0 * d0[x0, y0] + t1 * d0[x0, y1]) +
         s1 * (t0 * d0[x1, y0] + t1 * d0[x1, y1]))

    return b

def angleWrap(theta):
    # wraps to between [0, 2pi]
    while theta < -np.pi:
        theta += 2*np.pi
    while theta > np.pi:
        theta -= 2*np.pi
    return theta

def angleToC(theta):
    # get rotation matrix from theta
    cosx, sinx = np.cos(theta), np.sin(theta)
    C = np.array([[cosx, -sinx], [sinx, cosx]])
    return C

def constructH(t, theta):
    H = np.zeros((3, 3))
    H[:2, :2] = angleToC(theta)
    H[:2, 2:3] = t.reshape(2, 1)
    H[2, 2] = 1
    return H

def inverseH(H):
    invH = np.empty((3, 3))
    invH[2, 2] = 1
    invH[:2, :2] = H[:2, :2].T
    invH[:2, 2:3] = -H[:2, :2].T @ H[:2, 2:3]
    return invH

def orthogonal(v):
    return np.array([-v[1], v[0]])

def close(a,b):
    return abs(a-b) < 1E-7

def wcrossC(w, C):
    # for 2D, w cross C is [-wsin(theta), -wcos(theta); wcos(theta), -wsin(theta)]
    return w * np.array([[-C[1, 0], -C[0, 0]], [C[0, 0], -C[1, 0]]])
