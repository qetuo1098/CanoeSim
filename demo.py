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

try:
    from OpenGL.GLUT import *
    from OpenGL.GL import *
    from OpenGL.GLU import *
except ImportError:
    print('ERROR: PyOpenGL not installed properly.')
    sys.exit()

# Demo objects
class DrawStyle(enum.Enum):
    DYE_DENSITY = enum.auto()
    FLOW_VELOCITY = enum.auto()


@dataclass
class MousePose:
    x: float
    y: float


@dataclass
class Window:
    width_x: float
    height_y: float
    res: float
    size: float = field(init=False)

    def __post_init__(self):
        self.size = self.res + 2

# main code
draw_style = DrawStyle.DYE_DENSITY

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
boat = Boat(tf, (2, 6), 300, Pose(40, 10, 0), vel=Pose(2, 2, 0.5))  # n was 300
controller = Controller(boat)
# boat = Boat((3, 9), 300, Pose(40, 40, 7*pi/6))
counter = 0


def clear_data():
    """clear_data."""

    global dens, dens_new_source, window, vel, vel_new_source

    size = window.size
    vel.u[0:size, 0:size] = 0.0
    vel.v[0:size, 0:size] = 0.0
    vel_new_source.u[0:size, 0:size] = 0.0
    vel_new_source.v[0:size, 0:size] = 0.0
    dens[0:size, 0:size] = 0.0
    dens_new_source[0:size, 0:size] = 0.0


def pre_display():
    """pre_display."""

    glViewport(0, 0, window.width_x, window.height_y)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluOrtho2D(0.0, 1.0, 0.0, 1.0)
    glClearColor(0.0, 0.0, 0.0, 1.0)
    glClear(GL_COLOR_BUFFER_BIT)


def post_display():
    """post_display."""

    glutSwapBuffers()

def draw_boat():
    h = 1.0 / window.res

    # boat
    glColor3f(1.0, 1.0, 1.0)
    glPointSize(5.0)

    glBegin(GL_POINTS)
    for point in boat.tf.getTransformedPoses(boat.canoe_frame, boat.tf.root)[0].T:
        glVertex2f((point[0]-0.5)*h, (point[1]-0.5)*h)
    glEnd()

    # paddles
    glColor3f(0.5, 0.5, 1.0)
    glPointSize(5.0)

    glBegin(GL_POINTS)
    for paddle in boat.all_paddle_list:
        for point in boat.tf.getTransformedPoses(paddle.frame, boat.tf.root)[0].T:
            glVertex2f((point[0] - 0.5) * h, (point[1] - 0.5) * h)
    glEnd()


def draw_velocity():
    """draw_velocity."""

    h = 1.0 / window.res

    glColor3f(1.0, 1.0, 1.0)
    glLineWidth(1.0)

    glBegin(GL_LINES)
    for i in range(1, window.res + 1):
        x = (i - 0.5) * h
        for j in range(1, window.res + 1):
            y = (j - 0.5) * h
            glColor3f(1, 0, 0)
            glVertex2f(x, y)
            glVertex2f(x + vel.u[i, j], y + vel.v[i, j])
    glEnd()


def draw_density():
    """draw_density."""

    h = 1.0 / window.res

    glBegin(GL_QUADS)
    for i in range(0, window.res + 1):
        x = (i - 0.5) * h
        for j in range(0, window.res + 1):
            y = (j - 0.5) * h
            d00 = dens[i, j]
            d01 = dens[i, j + 1]
            d10 = dens[i + 1, j]
            d11 = dens[i + 1, j + 1]

            glColor3f(d00, d00, d00)
            glVertex2f(x, y)
            glColor3f(d10, d10, d10)
            glVertex2f(x + h, y)
            glColor3f(d11, d11, d11)
            glVertex2f(x + h, y + h)
            glColor3f(d01, d01, d01)
            glVertex2f(x, y + h)
    glEnd()


def get_from_UI(d, vel):
    """get_from_UI."""

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

    if mouse_down[GLUT_RIGHT_BUTTON]:
        d[i, j] = source

    old_mouse_pose = curr_mouse_pose


def key_func(key, mouse_x, mouse_y):
    """key_func."""

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
    """mouse_func."""

    global old_mouse_pose, curr_mouse_pose, mouse_down

    mouse_pose = MousePose(mouse_x, mouse_y)
    old_mouse_pose = curr_mouse_pose = mouse_pose
    mouse_down[button] = (state == GLUT_DOWN)


def motion_func(x, y):
    """motion_func."""

    global curr_mouse_pose

    curr_mouse_pose = MousePose(x, y)


def reshape_func(width, height):
    """reshape_func."""

    global win_x, win_y

    glutReshapeWindow(width, height)
    win_x = width
    win_y = height


def idle_func():
    """idle_func."""

    global dens, dens_new_source, window, visc, dt, diff, vel, vel_new_source
    global boat, counter

    get_from_UI(dens_new_source, vel_new_source)
    vel_step(window.res, vel, vel_new_source, visc, dt, boat)
    dens_step(window.res, dens, dens_new_source, vel, diff, dt)

    counter += 1
    controller.control()
    boat.stepForward(vel, dt)
    if counter % 5 == 0:
        print("Force: ", boat.getWrenches(vel))
    glutPostRedisplay()


def display_func():
    """display_func."""

    pre_display()
    if draw_style == DrawStyle.FLOW_VELOCITY:
        draw_velocity()
    elif draw_style == DrawStyle.DYE_DENSITY:
        draw_density()
    draw_boat()
    post_display()


def open_glut_window():
    """open_glut_window."""

    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE)
    glutInitWindowPosition(0, 0)
    glutInitWindowSize(window.width_x, window.height_y)
    glutCreateWindow("Alias | wavefront (porting by Alberto Santini)")
    glClearColor(0.0, 0.0, 0.0, 1.0)
    glClear(GL_COLOR_BUFFER_BIT)
    glutSwapBuffers()
    glClear(GL_COLOR_BUFFER_BIT)
    glutSwapBuffers()

    pre_display()

    glutKeyboardFunc(key_func)
    glutMouseFunc(mouse_func)
    glutMotionFunc(motion_func)
    glutReshapeFunc(reshape_func)
    glutIdleFunc(idle_func)
    glutDisplayFunc(display_func)


def run():
    glutInit(sys.argv)
    clear_data()
    open_glut_window()
    glutMainLoop()


if __name__ == "__main__":
    run()
