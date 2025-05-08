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

El modelo 3D del brazo robótico no nos pertenece, lo puedes conseguir en el siguiente link: https://cults3d.com/es/modelo-3d/artilugios/brazo-robotico-arduino-diy-con-control-de-smartphone-2023

## Arquitectura hardware


### Componentes



## Arquitectura software


### Descripción de los módulos

Puedes acceder a una descripción más detallada de cada módulo en sus directorios:

- [Image Processing](ImageProcessingModule/README.md): Módulo destinado a la detección de los bloques mediante el modelo de Ultralytics YOLO. Dentro de este módulo se incluye también el preprocesamiento de las imágenes captadas por la cámara, con la finalidad de normalizarlas y obtener un mejor reconocimiento, y la detección de su posición 3D mediante un algoritmo de Depth Estimation.
- [Movement](MovementModule/README.md): Módulo destinado al cálculo de la trayectoria del robot sin colisiones y al algoritmo de apilamiento, el cual debe tener en cuenta la estabilidad de la torre en tiempo real.
- [User Interface](UserInterfaceModule/README.md): Aplicación que permita visualizar la retransmisión de la cámara en tiempo real, iniciar y parar el funcionamiento del robot y un modo manual en el que puedas elegir qué pieza quieres que agarre el brazo robótico. 

## Bibliografía 

formato adecuado: yolo, opencv, coppelia

## Licencia

