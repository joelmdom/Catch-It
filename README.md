# CatchIt!

## Descripción del proyecto

Nuestro brazo robótico **CatchIt!** es un robot de cuatro ejes con garra diseñado para tareas de *pick and place*. La principal funcionalidad de este robot consiste en un reconocimiento en tiempo real de bloques rectangulares de madera, cálculo de su posición exacta, recoger el cubo más accesible, y colocarlo formando la torre de jenga. 

En la creación de este proyecto, destacan los siguientes retos técnicos:
- **Cálculo de profundidad**: Usando una cámara 2D, conseguir estimar las posiciones tridimensionales de los bloques.
- **Algoritmo de apilamiento**: Diseño de un algoritmo que garantice la estabilidad de la torre en construcción y evite colapsos al agregar bloques.

## Instalación

> [!TIP]
> Puedes consultar las librerías que hemos usado para este proyecto en el fichero [requirements.txt](requirements.txt)

> [!CAUTION]
> He cambiado enrutamientos de archivos 

en la carpeta del programa

compilar

```docker build . -t catch-it```

ejecutar en primer plano

```docker-compose up```

ejecutar en segundo plano

```docker-compose up -d```

## Funcionamiento y demostración


## Piezas 3D

El modelo 3D del brazo robótico no nos pertenece, lo puedes conseguir aquí en el siguiente link: https://cults3d.com/es/modelo-3d/artilugios/brazo-robotico-arduino-diy-con-control-de-smartphone-2023

## Arquitectura hardware


### Componentes



## Arquitectura software


### Descripción de los módulos

Puedes acceder a una descripción más detallada de cada módulo en sus directorios:

- [Image Processing](ImageProcessingModule/README.md)
- [Movement](MovementModule/README.md)
- [User Interface](UserInterfaceModule/README.md)



## Bibliografía 

formato adecuado: yolo, opencv, coppelia

## Licencia

