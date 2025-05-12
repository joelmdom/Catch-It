"""
Programa de prueba: coloca los bloques para hacer la torre. el robot no los coloca simplemente spawnean.
para probar algoritmos de apilamiento y estabilidad.
"""

from coppelia_functions import *
import time
import sim
from math import radians

class BuilderModule:
    def __init__(self):
        # conectar a coppelia
        self.clientID = coppelia_connect(19999)

    def spawn_jenga_block(self, x, y, z, orientation_gamma):
        # copy paste block
        _, block = sim.simxGetObjectHandle(self.clientID, "Cuboid53", sim.simx_opmode_blocking)
        print(block)
        _, ret = sim.simxCopyPasteObjects(self.clientID, [block], sim.simx_opmode_blocking)
        print(block)
        block = ret[0]
        sim.simxSetObjectOrientation(self.clientID, block, block, [0,0,radians(orientation_gamma)], sim.simx_opmode_blocking)
        sim.simxSetObjectPosition(self.clientID, block, -1, [x, y, z], sim.simx_opmode_blocking)
        return block

    def build_tower_level_even(self, x,y,z):
        self.spawn_jenga_block(x, y-0.03, z, 0)
        self.spawn_jenga_block(x, y+0.00, z, 0)
        self.spawn_jenga_block(x, y+0.03, z, 0)
        return

    def build_tower_level_odd(self, x,y,z):
        self.spawn_jenga_block(x-0.03, y , z, 90)
        self.spawn_jenga_block(x+0.00, y, z, 90)
        self.spawn_jenga_block(x+0.03, y, z, 90)
        pass

    def build_tower(self, n_blocks, center):
        levels = n_blocks / 3
        levels = int(levels)
        print(levels)
        x,y,z = center
        z += 0.01 # para dejarla caer un poco y desequilibrar

        for level in range(levels):
            # self.build_tower_level_even(x, y, z)
            if level % 2 == 0:
                self.build_tower_level_even(x,y,z)
            else:
                self.build_tower_level_odd(x,y,z)
            z += 0.015


if __name__ == '__main__':
    builder = BuilderModule()
    x = -0.3
    y = -0.1
    z = 0.05
    # for i in range(100):
    #     builder.spawn_jenga_block(x,y,z)
    #     y -= 0.03
        # time.sleep(1)

    # builder.build_tower_level_even(x,y,z)
    builder.build_tower(30, [x,y,z])