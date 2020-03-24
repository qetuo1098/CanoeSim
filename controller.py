from boat import *

class Controller:
    """
    Drives the canoe to achieve a desired goal.
    Generates commands using information from the boat object, and operates the paddles by changing their angular velocities.
    """
    def __init__(self, boat):
        self.boat = boat

    def getCommand(self):
        return np.zeros(4)

    def control(self):
        # call this method to send a command to the boat
        command = self.getCommand()
        # print(command)
        self.boat.handleL.setAngularVel(command[0])
        self.boat.handleR.setAngularVel(command[1])
        self.boat.paddleL.setAngularVel(command[2])
        self.boat.paddleR.setAngularVel(command[3])


class OpenLoopController(Controller):
    def __init__(self, boat):
        super(OpenLoopController, self).__init__(boat)
        # open loop controller variables: hardcoded control commands
        self.counter = 0
        self.vel = 3
        self.interval = 20
        self.commands = np.vstack((
            np.repeat(np.array([[0, 0, 0, 0]]), self.interval, axis=0),
            np.repeat(np.array([[self.vel, -self.vel, -self.vel, self.vel]]), self.interval, axis=0),
            np.repeat(np.array([[-self.vel, self.vel, 0, 0]]), self.interval, axis=0),
            np.repeat(np.array([[0, 0, self.vel, -self.vel]]), self.interval, axis=0),
            np.repeat(np.array([[self.vel, -self.vel, -self.vel, self.vel]]), self.interval, axis=0),
            np.repeat(np.array([[-self.vel, self.vel, 0, 0]]), self.interval, axis=0),
            np.repeat(np.array([[0, 0, self.vel, -self.vel]]), self.interval/2, axis=0),
            np.repeat(np.array([[self.vel, -self.vel, -self.vel, self.vel]]), self.interval, axis=0),
            np.repeat(np.array([[0, self.vel, 0, 0]]), self.interval, axis=0),
            np.repeat(np.array([[-self.vel, -self.vel, 0, 0]]), self.interval, axis=0),
            np.repeat(np.array([[0, 0, 0, 0]]), self.interval*2, axis=0),
            np.repeat(np.array([[self.vel, self.vel, 0, 0]]), self.interval/2, axis=0)
        ))

    def getCommand(self):
        # generates a command
        if self.counter < len(self.commands):
            command = self.commands[self.counter]
            self.counter += 1
        else:
            command = np.zeros(4)
        return command

class EmptyController(Controller):
    def __init__(self, boat):
        super(EmptyController, self).__init__(boat)
    
    def control(self):
        pass

class JitterController(Controller):
    def __init__(self, boat):
        super(JitterController, self).__init__(boat)
        # open loop controller variables: hardcoded control commands
        self.vel = 3
        self.pos = True

    def getCommand(self):
        self.pos = not self.pos
        if self.pos:
            return np.array([self.vel, -self.vel, -self.vel, self.vel])
        else:
            return np.array([-self.vel, self.vel, self.vel, -self.vel])
