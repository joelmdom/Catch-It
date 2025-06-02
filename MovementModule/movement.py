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
        self.joints = [self.m1, self.m2, self.m3, self.m4, self.m5]
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

    def move_test(self):
        sim.simxSetJointTargetPosition(self.clientID, self.m1, 10, sim.simx_opmode_blocking)
        sim.simxSetJointTargetPosition(self.clientID, self.m2, 10, sim.simx_opmode_blocking)
        sim.simxSetJointTargetPosition(self.clientID, self.m3, 10, sim.simx_opmode_blocking)
        sim.simxSetJointTargetPosition(self.clientID, self.m4, 10, sim.simx_opmode_blocking)
        sim.simxSetJointTargetPosition(self.clientID, self.m5, 10, sim.simx_opmode_blocking)
        self.close_claw()

    def move_joint(self, joint_id, target):
        sim.simxSetJointTargetPosition(self.clientID, self.joints[joint_id-1], target, sim.simx_opmode_blocking)

    def open_claw(self):
        sim.simxSetJointTargetPosition(self.clientID, self.m6_l, radians(30), sim.simx_opmode_blocking)
        sim.simxSetJointTargetPosition(self.clientID, self.m6_r, radians(-30), sim.simx_opmode_blocking)

    def close_claw(self):
        sim.simxSetJointTargetPosition(self.clientID, self.m6_l, radians(0), sim.simx_opmode_blocking)
        sim.simxSetJointTargetPosition(self.clientID, self.m6_r, radians(0), sim.simx_opmode_blocking)

    def move_arm_to_position(self,x,y,z):
        """
        Inverse kinematic movement. Adaptado de sessió 12
        """
        import numpy as np
        import math

        # Dimensiones de los brazos
        H = 0.2  # altura de base mm LINK0
        b = 0.3  # longitud de brazo m  LINK1
        ab = 0.16  # longitud de antebrazo  LINK2
        m = 0.24  # longitud de muñequilla m LINK3+LINK4+PINZA - dimension hasta el dummy de la pinza

        # Colocamos cabeceo y giro de la pinza a 0 grados inicialmente
        cabGrados = 0  # cabeceo de la pinza Joint3
        Axis5 = 0  # Giro de la pinza Joint4 en grados

        # Con esos datos conseguimos los grados de los primero 4 ejes, pero la orientación del joint 3 (cabeceo) no se ha modificado

        cabRAD = cabGrados * np.pi / 180
        Axis1 = math.atan2(y, x)
        M = math.sqrt(pow(x, 2) + pow(y, 2))
        xprima = M
        yprima = z

        Afx = math.cos(cabRAD) * m
        B = xprima - Afx
        Afy = math.sin(cabRAD) * m
        A = yprima + Afy - H
        Hip = math.sqrt(pow(A, 2) + pow(B, 2))
        alfa = math.atan2(A, B)
        beta = math.acos((pow(b, 2) - pow(ab, 2) + pow(Hip, 2)) / (2 * b * Hip))
        Axis2 = alfa + beta
        gamma = math.acos((pow(b, 2) + pow(ab, 2) - pow(Hip, 2)) / (2 * b * ab))
        Axis3 = gamma
        Axis4 = 2 * np.pi - cabRAD - Axis2 - Axis3

        j1 = Axis1 * 180 / np.pi
        j2 = 90 - Axis2 * 180 / np.pi
        j3 = 180 - Axis3 * 180 / np.pi
        j4 = 180 - Axis4 * 180 / np.pi
        j5 = Axis5  # joint5  Se ha dado en grados inicialmente
        print(j1, j2, j3, j4, j5)

        self.move_joint(1, j1 * np.pi / 180)
        self.move_joint(2, j2 * np.pi / 180)
        self.move_joint(3, j3 * np.pi / 180)
        self.move_joint(4, j4 * np.pi / 180)
        self.move_joint(5, j5 * np.pi / 180)

if __name__ == '__main__':
    movement = MovementModule()
    movement.reset()
    time.sleep(1)
    # movement.open_claw()
    # time.sleep(1)
    # movement.close_claw()
    time.sleep(1)
    movement.move_arm_to_position(-0.3,0,0.05)




