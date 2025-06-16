"""
Modulo para controlar el brazo.
"""
import numpy as np

import sim
import time
from math import radians

from coppelia_functions import *

class MovementModule:
    def __init__(self):
        pass

    def reset(self):
        pass

    def move_test(self):
        pass

    def move_joint(self, joint_id, target):
        pass

    def open_claw(self):
        pass

    def close_claw(self):
        pass

    def get_arm_dimensions(self):
        return 0,0,0,0

    def inverse_kinematics(self, x,y,z):
        import numpy as np
        import math

        H, ab, b, m = self.get_arm_dimensions()  # sera distinto depende si usamos la clase de sim o real

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
        # print(j1, j2, j3, j4, j5)

        return j1, j2, j3, j4, j5

    def move_arm_to_position(self,x,y,z):
        """
        Inverse kinematic movement. Adaptado de sessió 12.
        funciona tanto para coppelia como para motores reales si usamos la clase correcta
        """
        j1, j2, j3, j4, j5 = self.inverse_kinematics(x,y,z)

        print(f"Joint 1 at {j1} degrees")
        print(f"Joint 2 at {j2} degrees")
        print(f"Joint 3 at {j3} degrees")
        print(f"Joint 4 at {j4} degrees")
        print(f"Joint 5 at {j5} degrees")

        self.move_joint(1, j1 * np.pi / 180)
        time.sleep(1)

        # alerta: nuestro joint 4 y 5 estan en el orden invertido respecto al ejemplo.
        self.move_joint(5, j4 * np.pi / 180)
        time.sleep(1)
        self.move_joint(4, j5 * np.pi / 180)
        time.sleep(1)

        self.move_joint(3, j3 * np.pi / 180)
        time.sleep(1)
        self.move_joint(2, j2 * np.pi / 180)
        time.sleep(1)

    def move_arm_to_position_sin_muneca(self,x,y,z):
        """
        Inverse kinematic movement. Adaptado de sessió 12.
        funciona tanto para coppelia como para motores reales si usamos la clase correcta
        """
        j1, j2, j3, j4, j5 = self.inverse_kinematics(x,y,z)

        print(f"Joint 1 at {j1} degrees")
        print(f"Joint 2 at {j2} degrees")
        print(f"Joint 3 at {j3} degrees")
        print(f"Joint 4 at {j4} degrees")
        print(f"Joint 5 at {j5} degrees")

        self.move_joint(1, j1 * np.pi / 180)
        time.sleep(1)

        # alerta: nuestro joint 4 y 5 estan en el orden invertido respecto al ejemplo.
        # self.move_joint(5, j4 * np.pi / 180)
        time.sleep(1)
        self.move_joint(4, j5 * np.pi / 180)
        time.sleep(1)

        self.move_joint(3, j3 * np.pi / 180)
        time.sleep(1)
        self.move_joint(2, j2 * np.pi / 180)
        time.sleep(1)

class MovementModuleSim(MovementModule):
    """
    clase con todo el codigo especifico para coppelia.
    """
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
        sim.simxSetJointTargetPosition(self.clientID, self.joints[joint_id - 1], target, sim.simx_opmode_blocking)

    def open_claw(self):
        sim.simxSetJointTargetPosition(self.clientID, self.m6_l, radians(60), sim.simx_opmode_blocking)
        sim.simxSetJointTargetPosition(self.clientID, self.m6_r, radians(-60), sim.simx_opmode_blocking)

    def close_claw(self):
        sim.simxSetJointTargetPosition(self.clientID, self.m6_l, radians(0), sim.simx_opmode_blocking)
        sim.simxSetJointTargetPosition(self.clientID, self.m6_r, radians(0), sim.simx_opmode_blocking)

    def get_arm_dimensions(self):
        # Dimensiones de los brazos (ejemplo)
        # H = 0.2  # altura de base mm LINK0
        # b = 0.3  # longitud de brazo m  LINK1
        # ab = 0.16  # longitud de antebrazo  LINK2
        # m = 0.24  # longitud de muñequilla m LINK3+LINK4+PINZA - dimension hasta el dummy de la pinza

        # Dimensiones de los brazos (coppelia nuestro)
        H = 0.07 + 0.07  # altura base + altura eje central
        ab = 0.1
        b = 0.1
        m = 0.05 + 0.07  # muñeca + pinza
        return H, ab, b, m

    def stick_object(self, objectName: str):
        """
        hacer que la pieza se quede agarrada a la pinza
        """
        import sim
        clientID = self.clientID
        _, holder = sim.simxGetObjectHandle(clientID, 'Pinza_R_Simple', sim.simx_opmode_blocking)
        _, obj = sim.simxGetObjectHandle(clientID, objectName, sim.simx_opmode_blocking)
        # res, cuboid = sim.simxGetObjectHandle(clientID, 'Cuboid', sim.simx_opmode_blocking)
        sim.simxSetObjectParent(clientID, obj, holder, True, sim.simx_opmode_blocking)
        # sim.setObjectInt32Parameter(clientID, cuboid, sim.shapeintparam_static, 1)
        sim.simxSetObjectIntParameter(clientID, obj, sim.sim_shapeintparam_static, 1, sim.simx_opmode_blocking)
        return

    def stick_object_ref(self, obj):
        """
        hacer que la pieza se quede agarrada a la pinza
        """
        import sim
        clientID = self.clientID
        _, holder = sim.simxGetObjectHandle(clientID, 'Pinza_R_Simple', sim.simx_opmode_blocking)
        # _, obj = sim.simxGetObjectHandle(clientID, objectName, sim.simx_opmode_blocking)
        # res, cuboid = sim.simxGetObjectHandle(clientID, 'Cuboid', sim.simx_opmode_blocking)
        sim.simxSetObjectParent(clientID, obj, holder, True, sim.simx_opmode_blocking)
        # sim.setObjectInt32Parameter(clientID, cuboid, sim.shapeintparam_static, 1)
        sim.simxSetObjectIntParameter(clientID, obj, sim.sim_shapeintparam_static, 1, sim.simx_opmode_blocking)
        return

    def unstick_object(self, objectName: str):
        """
        hacer que la pieza ya no esté agarrada a la pinza
        """
        import sim
        clientID = self.clientID
        # _, holder = sim.simxGetObjectHandle(clientID, 'Pinza_R_Simple', sim.simx_opmode_blocking)
        _, obj = sim.simxGetObjectHandle(clientID, objectName, sim.simx_opmode_blocking)
        # res, cuboid = sim.simxGetObjectHandle(clientID, 'Cuboid', sim.simx_opmode_blocking)
        sim.simxSetObjectParent(clientID, obj, -1, True, sim.simx_opmode_blocking)
        # sim.setObjectInt32Parameter(clientID, cuboid, sim.shapeintparam_static, 1)
        sim.simxSetObjectIntParameter(clientID, obj, sim.sim_shapeintparam_static, 0, sim.simx_opmode_blocking)
        return
    def unstick_object_ref(self, obj):
        """
        hacer que la pieza ya no esté agarrada a la pinza
        """
        import sim
        clientID = self.clientID
        # _, holder = sim.simxGetObjectHandle(clientID, 'Pinza_R_Simple', sim.simx_opmode_blocking)
        # _, obj = sim.simxGetObjectHandle(clientID, objectName, sim.simx_opmode_blocking)
        # res, cuboid = sim.simxGetObjectHandle(clientID, 'Cuboid', sim.simx_opmode_blocking)
        sim.simxSetObjectParent(clientID, obj, -1, True, sim.simx_opmode_blocking)
        # sim.setObjectInt32Parameter(clientID, cuboid, sim.shapeintparam_static, 1)
        sim.simxSetObjectIntParameter(clientID, obj, sim.sim_shapeintparam_static, 0, sim.simx_opmode_blocking)
        return


    def spawn_jenga_block(self, x, y, z, orientation_gamma):
        # copy paste block
        _, block = sim.simxGetObjectHandle(self.clientID, "pieza", sim.simx_opmode_blocking)
        # print(block)
        _, ret = sim.simxCopyPasteObjects(self.clientID, [block], sim.simx_opmode_blocking)
        # print(block)
        block = ret[0]
        sim.simxSetObjectOrientation(self.clientID, block, block, [0,0,radians(orientation_gamma)], sim.simx_opmode_blocking)
        sim.simxSetObjectPosition(self.clientID, block, -1, [x, y, z], sim.simx_opmode_blocking)
        return block

    def wait_for_block(self, orientation_gamma):
        """
        In coppelia sim we spawn the object instead of waiting. In real life we check with computer vision if the block
        is placed in the correct position and rotation.
        """
        b = self.spawn_jenga_block(-0.2,0.0,0.025, orientation_gamma)
        time.sleep(2)
        return b

    def pick_and_place(self, x_dest, y_dest, z_dest, orientation_gamma):
        self.reset()
        H, ab, b, m = self.get_arm_dimensions()

        # Pick
        movement.open_claw()
        block = movement.wait_for_block(orientation_gamma)
        movement.move_arm_to_position(-0.2, 0, 0.07)
        time.sleep(1)
        movement.stick_object_ref(block)

        # Place
        movement.reset()
        movement.open_claw()
        movement.move_arm_to_position(x_dest, y_dest, z_dest)
        movement.unstick_object_ref(block)
        time.sleep(1)


    def build_tower_level_odd(self, x,y,z):
        # self.spawn_jenga_block(x-0.03, y , z, 90)
        # self.spawn_jenga_block(x+0.00, y, z, 90)
        # self.spawn_jenga_block(x+0.03, y, z, 90)]
        self.pick_and_place(x - 0.03, y, z, 90)
        self.pick_and_place(x + 0.00, y, z, 90)
        self.pick_and_place(x + 0.03, y, z, 90)
        pass

    def build_tower_level_even(self, x,y,z):
        # self.pick_and_place(x, y-0.03, z, 0)
        # self.pick_and_place(x, y+0.00, z, 0)
        # self.pick_and_place(x, y+0.03, z, 0)
        self.pick_and_place(x, y - 0.032, z, 0)
        self.pick_and_place(x, y + 0.000, z, 0)
        self.pick_and_place(x, y + 0.032, z, 0)
        pass

    def build_tower(self, n_blocks, center):
        levels = n_blocks / 3
        levels = int(levels)
        print(levels)
        x,y,z = center
        # z += 0.1 # para dejarla caer un poco y desequilibrar
        # z += 0.07 # para dejarla caer un poco y desequilibrar

        for level in range(levels):
            # self.build_tower_level_even(x, y, z)
            if level % 2 == 0:
                self.build_tower_level_even(x,y,z)
            else:
                self.build_tower_level_odd(x,y,z)
            z += 0.033


class MovementModuleReal(MovementModule):
    def __init__(self):
        import board
        from adafruit_motor import servo
        from adafruit_pca9685 import PCA9685
        i2c = board.I2C()
        pca = PCA9685(i2c)
        pca.frequency = 50

        # init all servos
        self.servos = []
        # for i in range(16):
        #     self.servos.append( servo.Servo(pca.channels[i]))

        self.servos.append(servo.Servo(pca.channels[0]))
        self.servos.append(servo.Servo(pca.channels[3]))
        self.servos.append(servo.Servo(pca.channels[6]))
        self.servos.append(servo.Servo(pca.channels[9]))
        self.servos.append(servo.Servo(pca.channels[12]))
        self.servos.append(servo.Servo(pca.channels[15]))

    def open_claw(self):
        self.servos[5].angle = 90

    def close_claw(self):
        self.servos[5].angle = 62

    def move_joint(self, joint_id, target):
        # pasar de radianes a grados
        #target = target / np.pi

        self.servos[joint_id].angle = target
        time.sleep(1)

    def reset(self):
        self.servos[0].angle = 0
        self.servos[1].angle = 0
        self.servos[2].angle = 0
        self.servos[3].angle = 0
        self.servos[4].angle = 0
#        self.servos[5].angle = 0

    def get_arm_dimensions(self):
        # Dimensiones de los brazos (coppelia nuestro)
        H = 0.13
        ab = 0.125
        b = 0.13
        m = 0.13
        return H, ab, b, m

if __name__ == '__main__':
    movement = MovementModuleSim()

    movement.build_tower(99, [0.2, 0.0, 0.07])

# lo dejo aqui por si acaso
    # movement.reset()
    # time.sleep(1)
    # movement.open_claw()
    # time.sleep(1)
    # movement.close_claw()
    # time.sleep(1)
    # movement.move_arm_to_position(0.3,0,0.520)
    # movement.move_arm_to_position(-0.22,0,0.15)

    # H, ab, b, m = movement.get_arm_dimensions()

    # movement.reset()
#    time.sleep(2)
#     movement.open_claw()
#     block = movement.wait_for_block(True)
#     movement.move_arm_to_position(-0.2, 0, 0.07)
#     time.sleep(1)
#     movement.stick_object_ref(block)
#
#     movement.reset()
#     movement.open_claw()
#     movement.move_arm_to_position(0.2, 0.0, 0.07)
#     movement.unstick_object_ref(block)
    # movement.rotate_90()
# movement.move_joint(4, 90)
#    movement.open_claw()
#    time.sleep(5)
#    movement.close_claw()
#    movement.move_joint(4, 180)
#     movement.move_joint(2,-30 * np.pi / 180)
#     movement.move_joint(3,-135 * np.pi / 180)
    # movement.open_claw()
    # movement.reset()
    # time.sleep(1)

    # movement.move_joint(4,90)

    # movement.move_arm_to_position_sin_muneca(-0.25, -0.1, 0.07)






