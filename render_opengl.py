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

class DrawStyle(enum.Enum):
    DYE_DENSITY = enum.auto()
    FLOW_VELOCITY = enum.auto()

@dataclass
class Window:
    width_x: float
    height_y: float
    res: float
    size: float = field(init=False)

    def __post_init__(self):
        self.size = self.res + 2


def pre_display(window):
    glViewport(0, 0, window.width_x, window.height_y)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluOrtho2D(0.0, 1.0, 0.0, 1.0)
    glClearColor(0.0, 0.0, 0.0, 1.0)
    glClear(GL_COLOR_BUFFER_BIT)


def post_display():
    glutSwapBuffers()

def draw_boat(window, boat):
    h = 1.0 / window.res

    # boat
    glColor3f(1.0, 1.0, 1.0)
    glPointSize(3.0)

    glBegin(GL_POINTS)
    for point in boat.tf.getTransformedPoses(boat.canoe_frame, boat.tf.root)[0].T:
        glVertex2f((point[0]-0.5)*h, (point[1]-0.5)*h)
    glEnd()

    # noneffective paddles
    glColor3f(0.5, 0.5, 1.0)
    glPointSize(3.0)

    glBegin(GL_POINTS)
    for paddle in boat.noneffective_paddle_set:
        for point in boat.tf.getTransformedPoses(paddle.frame, boat.tf.root)[0].T:
            glVertex2f((point[0] - 0.5) * h, (point[1] - 0.5) * h)
    glEnd()
    
    # effective paddles
    glColor3f(0.5, 1.0, 0.5)
    glPointSize(3.0)

    glBegin(GL_POINTS)
    for paddle in boat.effective_paddle_set:
        for point in boat.tf.getTransformedPoses(paddle.frame, boat.tf.root)[0].T:
            glVertex2f((point[0] - 0.5) * h, (point[1] - 0.5) * h)
    glEnd()


def draw_velocity(window, vel):
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

def draw_density(window, dens):
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

def draw_target(window, target_pose):
    h = 1.0 / window.res
    glColor3f(1.0, 0.5, 0.5)
    glPointSize(6.0)

    glBegin(GL_POINTS)
    glVertex2f((target_pose.point[0]-0.5)*h, (target_pose.point[1]-0.5)*h)
    glEnd()

def display_func(window, field, boat, target_pose=Pose(10, 10, 0), draw_style=DrawStyle.FLOW_VELOCITY):
    pre_display(window)
    if draw_style == DrawStyle.FLOW_VELOCITY:
        draw_velocity(window, field)
    elif draw_style == DrawStyle.DYE_DENSITY:
        draw_density(window, field)
    draw_boat(window, boat)
    draw_target(window, target_pose)
    post_display()


def open_glut_window(width_x=900, height_y=900, res=64):
    window = Window(width_x, height_y, res)
    glutInit()
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE)
    glutInitWindowPosition(0, 0)
    glutInitWindowSize(window.width_x, window.height_y)
    glutCreateWindow("OpenGL_render")
    glClearColor(0.0, 0.0, 0.0, 1.0)
    glClear(GL_COLOR_BUFFER_BIT)
    glutSwapBuffers()
    glClear(GL_COLOR_BUFFER_BIT)
    glutSwapBuffers()
    return window
