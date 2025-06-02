# Modulo de movimiento

## movement.py

Controla todo el movimiento del robot. Contiene metodos para mover los motores, abrir y cerrar la garra, y mover 
el brazo a la posición deseada (inverse kinematics)

Hay que importar la clase `MovementModuleSim` o `MovementModuleReal` dependiendo de si usamos el robot simulado en
Coppelia o la versión real. 

```python
from movement import MovementModuleSim

movement = MovementModuleSim()
movement.reset()
movement.move_arm_to_position(-0.3, 0, 0.10)
```

```python
from movement import MovementModuleReal

movement = MovementModuleReal()
movement.reset()
...
```
## stability.py

Algoritmo para comprobar el angulo de la torre y su estabilidad.

## place_blocks_test

Algoritmo de construcción de torre. De momento los bloques spawnean en el coppelia directamente, pero hay que integrarlo
con el modulo de movimiento para que sea el brazo robotico el que haga pick and place.

## coppelia_functions.py

Varias funciones para interactuar con copelia. Lo usa el modulo de movimiento internamente.
