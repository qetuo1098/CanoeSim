import numpy as np

def bilinear_interp(d0, x, y):

    x0, x1 = int(x), int(x) + 1
    y0, y1 = int(y), int(y) + 1
    s0, s1 = x1 - x, x - x0
    t0, t1 = y1 - y, y - y0
    b = (s0 * (t0 * d0[x0, y0] + t1 * d0[x0, y1]) +
         s1 * (t0 * d0[x1, y0] + t1 * d0[x1, y1]))

    return b

def angleWrap(theta):
    # wraps to between [0, 2pi]
    while theta < 0:
        theta += 2*np.pi
    while theta > 2*np.pi:
        theta -= 2*np.pi
    return theta

def angleToC(theta):
    cosx, sinx = np.cos(theta), np.sin(theta)
    C = np.array([[cosx, -sinx], [sinx, cosx]])
    return C