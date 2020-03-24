import numpy as np

class Controller:
    def __init__(self):
        self.counter = 0
    
    def getCommand(self):
        raise NotImplementedError("getCommand() not implemented for base class Controller")
    
    def outputCommand(self):
        command = self.getCommand()
        self.counter += 1
        return command

class OpenLoopController(Controller):
    def __init__(self):
        super(OpenLoopController, self).__init__()
        # open loop controller variables: hardcoded control commands
        self.interval = 14
        # self.all_paddle_list = [self.handleL, self.handleR, self.paddleL, self.paddleR]
        self.commands = np.vstack((
            np.repeat(np.array([[0, 0, 0, 0.]]), self.interval, axis=0),
            np.repeat(np.array([[1., -1., -1., -1.]]), 10, axis=0),
            np.repeat(np.array([[1, 0, 0, 0.]]), self.interval*7, axis=0),
            np.repeat(np.array([[-1, 0, 0, 0.]]), self.interval, axis=0),
            
            np.repeat(np.array([[-1., 1., 0, 0]]), self.interval, axis=0),
            np.repeat(np.array([[0, -1, 1., 1.]]), 13, axis=0),
            np.repeat(np.array([[1., -1., -1., -1.]]), self.interval, axis=0),
            np.repeat(np.array([[-1., 1., 0, 0]]), self.interval, axis=0),
            np.repeat(np.array([[0, -1, 1., 1.]]), 13, axis=0),
            np.repeat(np.array([[1., -1., -1., -1.]]), self.interval, axis=0),
            np.repeat(np.array([[-1., 1., 0, 0]]), self.interval, axis=0),
            np.repeat(np.array([[0, -1, 1., 1.]]), 13, axis=0),
            np.repeat(np.array([[1., -1., -1., -1.]]), self.interval, axis=0),
            np.repeat(np.array([[-1., 1., 0, 0]]), self.interval, axis=0),
            np.repeat(np.array([[0, -1, 1., 1.]]), 13, axis=0),
            np.repeat(np.array([[1., -1., -1., -1.]]), self.interval, axis=0),
            np.repeat(np.array([[-1., 1., 0, 0]]), self.interval, axis=0),
            np.repeat(np.array([[0, -1, 1., 1.]]), 13, axis=0),
            np.repeat(np.array([[1., -1., -1., -1.]]), self.interval, axis=0),
            np.repeat(np.array([[-1., 1., 0, 0]]), self.interval, axis=0),
            np.repeat(np.array([[0, -1, 1., 1.]]), 13, axis=0),
            np.repeat(np.array([[1., -1., -1., -1.]]), self.interval, axis=0),

            np.repeat(np.array([[0, 1., 0, 0]]), self.interval, axis=0),
            np.repeat(np.array([[-1., -1., 0, 0]]), self.interval*1.5, axis=0),
            np.repeat(np.array([[0, 0, 0, 0.]]), self.interval*2, axis=0),
            np.repeat(np.array([[1., 1., 0, 0]]), self.interval/1.5, axis=0)
        ))

    def getCommand(self):
        # generates a command
        if self.counter < len(self.commands):
            command = self.commands[self.counter]
        else:
            command = np.zeros(4)
        return command

class EmptyController(Controller):
    def __init__(self):
        super(EmptyController, self).__init__()
    
    def getCommand(self):
        return np.zeros(4)

class JitterController(Controller):
    def __init__(self):
        super(JitterController, self).__init__()
        # open loop controller variables: hardcoded control commands
        self.pos = True

    def getCommand(self):
        self.pos = not self.pos
        if self.pos:
            return np.array([1., -1., -1., 1.])
        else:
            return np.array([-1., 1., 1., -1.])
