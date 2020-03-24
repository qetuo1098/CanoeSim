"""Real-Time Fluid Dynamics for Games by Jos Stam (2003).

Parts of author's work are also protected
under U. S. patent #6,266,071 B1 [Patent].
"""
from boat import *


def set_bnd(N, b, x):
    """We assume that the fluid is contained in a box with solid walls.

    No flow should exit the walls. This simply means that the horizontal
    component of the velocity should be zero on the vertical walls, while the
    vertical component of the velocity should be zero on the horizontal walls.
    For the density and other fields considered in the code we simply assume
    continuity. The following code implements these conditions.
    b = 0: continuity (used for dye density)
    b = 1: horizontal vel reflection
    b = 2: vertical vel reflection
    x: density/velocity field
    """

    for i in range(1, N + 1):
        if b == 1:
            x[0, i] = -x[1, i]
            x[N + 1, i] = -x[N, i]
        else:
            x[0, i] = x[1, i]
            x[N + 1, i] = x[N, i]
        if b == 2:
            x[i, 0] = -x[i, 1]
            x[i, N + 1] = -x[i, N]
        else:
            x[i, 0] = x[i, 1]
            x[i, N + 1] = x[i, N]

    # set corners
    x[0, 0] = 0.5 * (x[1, 0] + x[0, 1])
    x[0, N + 1] = 0.5 * (x[1, N + 1] + x[0, N])
    x[N + 1, 0] = 0.5 * (x[N, 0] + x[N + 1, 1])
    x[N + 1, N + 1] = 0.5 * (x[N, N + 1] + x[N + 1, N])


def lin_solve(N, b, x, x0, a, c):
    """lin_solve."""

    for k in range(0, 20):
        x[1:N + 1, 1:N + 1] = (x0[1:N + 1, 1:N + 1] + a *
                               (x[0:N, 1:N + 1] +
                                x[2:N + 2, 1:N + 1] +
                                x[1:N + 1, 0:N] +
                                x[1:N + 1, 2:N + 2])) / c
        set_bnd(N, b, x)


def add_source(N, x, s, dt):
    """Addition of forces: the density increases due to sources."""

    size = (N + 2)
    x[0:size, 0:size] += dt * s[0:size, 0:size]


def diffuse(N, b, x, x0, diff, dt):
    """Diffusion: the density diffuses at a certain rate.

    The basic idea behind our method is to find the densities which when
    diffused backward in time yield the densities we started with. The simplest
    iterative solver which works well in practice is Gauss-Seidel relaxation.
    """

    a = dt * diff * N * N
    lin_solve(N, b, x, x0, a, 1 + 4 * a)


def advect(N, b, d, d0, follow_vel, dt):
    """Advection: the density follows the velocity field.

    The basic idea behind the advection step. Instead of moving the cell
    centers forward in time through the velocity field, we look for the
    particles which end up exactly at the cell centers by tracing backwards in
    time from the cell centers.
    """

    dt0 = dt * N
    for i in range(1, N + 1):
        for j in range(1, N + 1):
            # project backwards through time
            x = i - dt0 * follow_vel.u[i, j]
            y = j - dt0 * follow_vel.v[i, j]

            # bound density to be within the window
            x = min(max(x, 0.5), N + 0.5)
            y = min(max(y, 0.5), N + 0.5)

            d[i, j] = bilinear_interp(d0, x, y)

    set_bnd(N, b, d)


def project(N, vel, vel_new_source):
    """project."""
    u = vel.u
    v = vel.v
    p = vel_new_source.u
    div = vel_new_source.v
    h = 1.0 / N
    div[1:N + 1, 1:N + 1] = (-0.5 * h *
                             (u[2:N + 2, 1:N + 1] - u[0:N, 1:N + 1] +
                              v[1:N + 1, 2:N + 2] - v[1:N + 1, 0:N]))
    p[1:N + 1, 1:N + 1] = 0
    set_bnd(N, 0, div)
    set_bnd(N, 0, p)
    lin_solve(N, 0, p, div, 1, 4)
    u[1:N + 1, 1:N + 1] -= 0.5 * (p[2:N + 2, 1:N + 1] - p[0:N, 1:N + 1]) / h
    v[1:N + 1, 1:N + 1] -= 0.5 * (p[1:N + 1, 2:N + 2] - p[1:N + 1, 0:N]) / h
    set_bnd(N, 1, u)
    set_bnd(N, 2, v)


def dens_step(N, x, x0, vel, diff, dt):
    """Evolving density.

    It implies advection, diffusion, addition of sources.
    """

    add_source(N, x, x0, dt)
    x0, x = x, x0  # swap
    diffuse(N, 0, x, x0, diff, dt)
    x0, x = x, x0  # swap
    advect(N, 0, x, x0, vel, dt)


def vel_step(N, vel, vel_new_source, visc, dt, boat):
    """Evolving velocity.

    It implies self-advection, viscous diffusion, addition of forces.
    """

    add_source(N, vel.u, vel_new_source.u, dt)
    add_source(N, vel.v, vel_new_source.v, dt)
    propelVelField(vel, boat)
    vel, vel_new_source = vel_new_source, vel

    diffuse(N, 1, vel.u, vel_new_source.u, visc, dt)
    diffuse(N, 2, vel.v, vel_new_source.v, visc, dt)
    project(N, vel, vel_new_source)
    vel, vel_new_source = vel_new_source, vel

    advect(N, 1, vel.u, vel_new_source.u, vel_new_source, dt)
    advect(N, 2, vel.v, vel_new_source.v, vel_new_source, dt)
    project(N, vel, vel_new_source)
