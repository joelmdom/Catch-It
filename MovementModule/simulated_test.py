"""
Programa de prueba: mover todos los motores del brazo en coppelia. Abrir y cerrar pinza. Falta inverse kinematics!!!
"""

# funciones del coppelia
import sim
def coppelia_connect(port):
    sim.simxFinish(-1) # just in case, close all opened connections
    clientID=sim.simxStart('127.0.0.1',port,True,True,2000,5) # Connect
    if clientID == -1: raise Exception("Error: could not connect to coppelia.")
    else: print("connected to", port)
    # if clientID == 0: print("connected to", port)
    # else: print("could not connect")
    return clientID

class MovementModule():
    def __init__(self):
        # conectar a coppelia
        self.clientID = coppelia_connect(19999)
        # print(self.clientID)

        # get all servomotors
        retCode, self.m1 = sim.simxGetObjectHandle(self.clientID, 'joint1', sim.simx_opmode_blocking)
        retCode, self.m2 = sim.simxGetObjectHandle(self.clientID, 'joint2', sim.simx_opmode_blocking)
        retCode, self.m3 = sim.simxGetObjectHandle(self.clientID, 'joint3', sim.simx_opmode_blocking)
        retCode, self.m4 = sim.simxGetObjectHandle(self.clientID, 'joint4', sim.simx_opmode_blocking)
        retCode, self.m5 = sim.simxGetObjectHandle(self.clientID, 'joint5', sim.simx_opmode_blocking)

        # pinza: en el brazo real solo habrá un motor porque va con engranaje y no sé cómo simularlo en coppelia
        retCode, self.m6_l = sim.simxGetObjectHandle(self.clientID, 'joint6_l', sim.simx_opmode_blocking)
        retCode, self.m6_r = sim.simxGetObjectHandle(self.clientID, 'joint6_r', sim.simx_opmode_blocking)

    def move(self):
        # sim.simxSetJointTargetPosition(self.clientID, self.m1, 40, sim.simx_opmode_blocking)
        sim.simxSetJointTargetPosition(self.clientID, self.m2, 40, sim.simx_opmode_blocking)
        # sim.simxSetJointTargetPosition(self.clientID, self.m3, 40, sim.simx_opmode_blocking)
        # sim.simxSetJointTargetPosition(self.clientID, self.m4, 40, sim.simx_opmode_blocking)
        # sim.simxSetJointTargetPosition(self.clientID, self.m5, 40, sim.simx_opmode_blocking)
        # sim.simxSetJointTargetPosition(self.clientID, self.m6, 0, sim.simx_opmode_blocking)


if __name__ == '__main__':
    movement = MovementModule()
    movement.move()