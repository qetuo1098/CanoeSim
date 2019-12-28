import numpy as np
from sklearn.preprocessing import normalize

def bilinear_interp(d0, x, y):
    # bilinearly interpolate value at field d0[x,y], with x,y being floats
    if x < 0:
        x = 0
    if y < 0:
        y = 0
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
    """
    Construct a 2D rotation matrix from theta
    :param theta: scalar
    :return: 2x2 C
    """
    cosx, sinx = np.cos(theta), np.sin(theta)
    C = np.array([[cosx, -sinx], [sinx, cosx]])
    return C

def constructH(t, theta):
    """
    Construct a 2D homogeneous transformation matrix given t and theta: H = [C t; 0 1]
    :param t: 2x1 or 1x2
    :param theta: scalar
    :return: 3x3 H
    """
    H = np.zeros((3, 3))
    H[:2, :2] = angleToC(theta)
    H[:2, 2:3] = t.reshape(2, 1)
    H[2, 2] = 1
    return H

def inverseH(H):
    """
    Finds the inverse of a 2D homogeneous transformation matrix more efficiently than (but equivalent to) using
    np.linalg.inv(H). For H = [C t; 0 1], inv(H) = [C.T -C.T@t; 0 1].
    :param H: 3x3 to be inverted
    :return: 3x3 inverse of H
    """
    invH = np.empty((3, 3))
    invH[2, 2] = 1
    invH[:2, :2] = H[:2, :2].T
    invH[:2, 2:3] = -H[:2, :2].T @ H[:2, 2:3]
    return invH

def orthogonal(v):
    # Finds the CW orthogonal vector to v. For 2D, CW orthogonal of [v0; v1] is [-v1; v0]
    return np.array([-v[1], v[0]])

def close(a,b):
    # True if a is close to b, False otherwise
    return abs(a-b) < 1E-7

def wcrossC(w, C):
    # for 2D, w cross C is [-wsin(theta), -wcos(theta); wcos(theta), -wsin(theta)]
    return w * np.array([[-C[1, 0], -C[0, 0]], [C[0, 0], -C[1, 0]]])

def convertVelToForce(v, n):
    """
    Project v onto n, then square the magnitude. Formula: sign(v dot n) * (v dot n)^2 * n
    :param v: velocity, 1x2
    :param n: unit normal vector, 1x2
    :return: f: force, 1x2
    """
    v = v.reshape(2)
    n = n.reshape(2)
    vdotn = v.dot(n)
    return np.sign(vdotn) * np.abs(vdotn**2) * n  # switch to v^2 when border is fixed
