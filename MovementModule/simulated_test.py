"""
Programa de prueba: mover todos los motores del brazo en coppelia. Abrir y cerrar pinza. Falta inverse kinematics!!!
"""

import sim
import time
from math import radians

from coppelia_functions import *

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

    def reset(self):
        sim.simxSetJointTargetPosition(self.clientID, self.m1, 0, sim.simx_opmode_blocking)
        sim.simxSetJointTargetPosition(self.clientID, self.m2, 0, sim.simx_opmode_blocking)
        sim.simxSetJointTargetPosition(self.clientID, self.m3, 0, sim.simx_opmode_blocking)
        sim.simxSetJointTargetPosition(self.clientID, self.m4, 0, sim.simx_opmode_blocking)
        sim.simxSetJointTargetPosition(self.clientID, self.m5, 0, sim.simx_opmode_blocking)
        self.close_claw()

    def move(self):
        sim.simxSetJointTargetPosition(self.clientID, self.m1, 10, sim.simx_opmode_blocking)
        sim.simxSetJointTargetPosition(self.clientID, self.m2, 10, sim.simx_opmode_blocking)
        sim.simxSetJointTargetPosition(self.clientID, self.m3, 10, sim.simx_opmode_blocking)
        sim.simxSetJointTargetPosition(self.clientID, self.m4, 10, sim.simx_opmode_blocking)
        sim.simxSetJointTargetPosition(self.clientID, self.m5, 10, sim.simx_opmode_blocking)
        self.close_claw()

    def open_claw(self):
        sim.simxSetJointTargetPosition(self.clientID, self.m6_l, radians(30), sim.simx_opmode_blocking)
        sim.simxSetJointTargetPosition(self.clientID, self.m6_r, radians(-30), sim.simx_opmode_blocking)

    def close_claw(self):
        sim.simxSetJointTargetPosition(self.clientID, self.m6_l, radians(0), sim.simx_opmode_blocking)
        sim.simxSetJointTargetPosition(self.clientID, self.m6_r, radians(0), sim.simx_opmode_blocking)

if __name__ == '__main__':
    movement = MovementModule()
    movement.reset()
    time.sleep(1)
    movement.open_claw()
    time.sleep(1)
    movement.close_claw()
    time.sleep(1)
    movement.move()