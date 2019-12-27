from boat import *

class Controller:
    def __init__(self, boat):
        self.boat = boat
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
            np.repeat(np.array([[0, 0, self.vel, -self.vel]]), self.interval, axis=0)
        ))

    def getCommand(self):
        if self.counter < len(self.commands):
            command = self.commands[self.counter]
            self.counter += 1
        else:
            command = np.zeros(4)
        return command

    def control(self):
        command = self.getCommand()
        self.boat.handleL.setAngularVel(command[0])
        self.boat.handleR.setAngularVel(command[1])
        self.boat.paddleL.setAngularVel(command[2])
        self.boat.paddleR.setAngularVel(command[3])
