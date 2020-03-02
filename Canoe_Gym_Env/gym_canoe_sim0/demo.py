"""Real-Time Fluid Dynamics for Games by Jos Stam (2003).

Parts of author's work are also protected
under U. S. patent #6,266,071 B1 [Patent].

Original paper by Jos Stam, "Real-Time Fluid Dynamics for Games".
Proceedings of the Game Developer Conference, March 2003

http://www.dgp.toronto.edu/people/stam/reality/Research/pub.html

Tested on
  python 2.4
  numarray 1.1.1
  PyOpenGL-2.0.2.01.py2.4-numpy23
  glut-3.7.6

How to use this demo:
  Add densities with the right mouse button
  Add velocities with the left mouse button and dragging the mouse
  Toggle density/velocity display with the 'v' key
  Clear the simulation by pressing the 'c' key
"""

import sys
from solver_c import *
from controller import *
import render_opengl as gl
from render_opengl import DrawStyle, Window
try:
    from OpenGL.GLUT import *
    from OpenGL.GL import *
    from OpenGL.GLU import *
except ImportError:
    print('ERROR: PyOpenGL not installed properly.')
    sys.exit()

# Demo objects
@dataclass
class MousePose:
    x: float
    y: float

# main code
draw_style = DrawStyle.FLOW_VELOCITY

dt = 0.05
diff = 0.0
visc = 0.005
force = 5.0
source = 100.0

window = Window(width_x=900, height_y=900, res=64)

old_mouse_pose = MousePose(0, 0)
curr_mouse_pose = MousePose(0, 0)

# mouse_down format: left, middle, right, scroll_up, scroll_down
mouse_down = [False, False, False, False, False]

""" Start with two grids.
One that contains the density values from the previous time step and one that
will contain the new values. For each grid cell of the latter we trace the
cell's center position backwards through the velocity field. We then linearly
interpolate from the grid of previous density values and assign this value to
the current grid cell.
"""
vel = VelField(window.size, window.size)
vel_new_source = copy(vel)
dens = np.zeros((window.size, window.size), float64)  # density
dens_new_source = np.zeros((window.size, window.size), float64)

tf = TransformTree()
boat = Boat(tf, (1, 5), 300/4, Pose(30, 15, 0), vel=Pose(0, 0, 0))  # n was 300
print("system args:", sys.argv)

if len(sys.argv) > 1 and sys.argv[1] == '1':
    controller = OpenLoopController(boat)
else:
    controller = EmptyController(boat)
# boat = Boat((3, 9), 300, Pose(40, 40, 7*pi/6))
counter = 0

in_bounds = True

def clear_data():
    global dens, dens_new_source, window, vel, vel_new_source

    size = window.size
    vel.u[0:size, 0:size] = 0.0
    vel.v[0:size, 0:size] = 0.0
    vel_new_source.u[0:size, 0:size] = 0.0
    vel_new_source.v[0:size, 0:size] = 0.0
    dens[0:size, 0:size] = 0.0
    dens_new_source[0:size, 0:size] = 0.0


def get_from_UI(d, vel):
    global old_mouse_pose

    d[0:window.size, 0:window.size] = 0.0
    vel.u[0:window.size, 0:window.size] = 0.0
    vel.v[0:window.size, 0:window.size] = 0.0

    if not mouse_down[GLUT_LEFT_BUTTON] and not mouse_down[GLUT_RIGHT_BUTTON]:
        return

    i = int((curr_mouse_pose.x / float(window.width_x)) * window.res + 1)
    j = int(((window.height_y - float(curr_mouse_pose.y)) / float(window.height_y)) * float(window.res) + 1.0)

    if i < 1 or i > window.res or j < 1 or j > window.res:
        return

    if mouse_down[GLUT_LEFT_BUTTON]:
        vel.u[i, j] = force * (curr_mouse_pose.x - old_mouse_pose.x)
        vel.v[i, j] = force * (old_mouse_pose.y - curr_mouse_pose.y)
        # print(vel.u[i, j], vel.v[i, j])

    if mouse_down[GLUT_RIGHT_BUTTON]:
        d[i, j] = source

    old_mouse_pose = curr_mouse_pose

    print("Mouse pose:", i, j)


def key_func(key, mouse_x, mouse_y):
    global draw_style
    mouse_pose = MousePose(mouse_x, mouse_y)  # unused
    if key == b'c' or key == b'C':
        clear_data()
    if key == b'v' or key == b'V':
        if draw_style == DrawStyle.FLOW_VELOCITY:
            draw_style = DrawStyle.DYE_DENSITY
        else:
            draw_style = DrawStyle.FLOW_VELOCITY
    if key == b'a':
        boat.moveByPose(Pose(-1, 0, 0))
    if key == b'd':
        boat.moveByPose(Pose(1, 0, 0))
    if key == b'w':
        boat.moveByPose(Pose(0, 1, 0))
    if key == b's':
        boat.moveByPose(Pose(0, -1, 0))
    if key == b'q':
        boat.moveByPose(Pose(0, 0, 0.1))
    if key == b'e':
        boat.moveByPose(Pose(0, 0, -0.1))
    if key == b'1':
        boat.handleL.setAngularVel(2.0)
    if key == b'2':
        boat.handleL.setAngularVel(-2.0)
    if key == b'3':
        boat.handleR.setAngularVel(2.0)
    if key == b'4':
        boat.handleR.setAngularVel(-2.0)
    if key == b'5':
        boat.paddleL.setAngularVel(2.0)
    if key == b'6':
        boat.paddleL.setAngularVel(-2.0)
    if key == b'7':
        boat.paddleR.setAngularVel(2.0)
    if key == b'8':
        boat.paddleR.setAngularVel(-2.0)
    if key not in (b'1', b'2', b'3', b'4', b'5', b'6', b'7', b'8'):
        boat.handleL.setAngularVel(0)
        boat.handleR.setAngularVel(0)
        boat.paddleL.setAngularVel(0)
        boat.paddleR.setAngularVel(0)


def mouse_func(button, state, mouse_x, mouse_y):
    global old_mouse_pose, curr_mouse_pose, mouse_down

    mouse_pose = MousePose(mouse_x, mouse_y)
    old_mouse_pose = curr_mouse_pose = mouse_pose
    mouse_down[button] = (state == GLUT_DOWN)


def motion_func(x, y):
    global curr_mouse_pose

    curr_mouse_pose = MousePose(x, y)


def reshape_func(width, height):
    global win_x, win_y

    glutReshapeWindow(width, height)
    win_x = width
    win_y = height


def idle_func():
    global dens, dens_new_source, window, visc, dt, diff, vel, vel_new_source
    global boat, counter, in_bounds

    if not in_bounds:
        exit("out of frame")

    get_from_UI(dens_new_source, vel_new_source)

    # add an artificial velocity source
    # for i in range(0, 10):
    #     vel_new_source.v[30+i, 20] = 0.5
    #     vel_new_source.u[30+i, 20] = 0.5

    vel_step(window.res, vel, vel_new_source, visc, dt, boat)
    dens_step(window.res, dens, dens_new_source, vel, diff, dt)

    counter += 1
    controller.control()
    in_bounds = boat.stepForward(vel, dt)
    print(in_bounds)
    if counter % 5 == 0:
        print("Force: ", boat.getWrenches(vel))
        print("Vel:", boat.vel.point, boat.vel.theta)
    glutPostRedisplay()


def display_func():
    if draw_style == DrawStyle.FLOW_VELOCITY:
        gl.display_func(window, vel, boat, draw_style)
    elif draw_style == DrawStyle.DYE_DENSITY:
        gl.display_func(window, dens, boat, draw_style)


def open_glut_window():
    gl.open_glut_window()
    gl.pre_display(window)
    glutKeyboardFunc(key_func)
    glutMouseFunc(mouse_func)
    glutMotionFunc(motion_func)
    glutReshapeFunc(reshape_func)
    glutIdleFunc(idle_func)
    glutDisplayFunc(display_func)


def run():
    clear_data()
    open_glut_window()
    glutMainLoop()


if __name__ == "__main__":
    run()
