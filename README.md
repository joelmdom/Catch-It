# CatchIt!

## Índice

- [Descripción del proyecto](#descripción-del-proyecto)
- [Instalación](#instalación)
- [Funcionamiento y demostración](#funcionamiento-y-demostración)
- [Piezas 3D](#piezas-3d)
- [Arquitectura hardware](#arquitectura-hardware)
     - [Componentes](#componentes)
- [Arquitectura software](#arquitectura-software)
     - [Descripción de los módulos](#descripción-de-los-módulos)
- [Bibliografía](#bibliografía)
- [Autores](#autores)
- [Licencia](#licencia)

## Descripción del proyecto

Nuestro brazo robótico **CatchIt!** es un robot de cinco ejes con garra diseñado para tareas de *pick and place*. La principal funcionalidad de este robot consiste en un reconocimiento en tiempo real de bloques rectangulares de madera, cálculo de su posición exacta, recoger el cubo más accesible, y colocarlo formando la torre de jenga. 

En la creación de este proyecto, destacan los siguientes retos técnicos:
- **Cálculo de profundidad**: Usando una cámara 2D, conseguir estimar las posiciones tridimensionales de los bloques.
- **Algoritmo de apilamiento**: Diseño de un algoritmo que garantice la estabilidad de la torre en construcción y evite colapsos al agregar bloques.

## Instalación

> [!TIP]
> Puedes consultar las librerías que hemos usado para este proyecto en el fichero [requirements.txt](requirements.txt)

En el directorio destinado al programa, compilar y ejecutar:

```docker build . -t catch-it -f ./Install/Dockerfile && docker-compose -f ./Install/docker-compose.yaml up```

## Funcionamiento y demostración

[Demo] (https://youtu.be/cBpDrhQ7hNY) 

## Piezas 3D

El modelo 3D del brazo robótico no nos pertenece, lo puedes conseguir en el siguiente link: https://cults3d.com/es/modelo-3d/artilugios/brazo-robotico-arduino-diy-con-control-de-smartphone-2023

## Arquitectura hardware
![Schematic_RLP_CatchIt](https://github.com/user-attachments/assets/83c8dc36-ed23-4169-a5e3-813045c7e362)



### Componentes

| Producto  | Descripción         | Imagen |
|-----------|---------------------|-------------|
| Raspberry Pi 4 Modelo B - 8GB RAM    | Placa SBC con procesador Broadcom BCM2711, conectividad Wifi 2.4 y 5.0 GHz y Bluetooth 5.0, conectividad ethernet, 4 puertos USB, GPIO de 40 pines. Requiere alimentador de 5.1V/3A con conector USB C     |     ![raspberry-pi-4-modelo-b-8gb-ram](https://github.com/user-attachments/assets/c90ecbba-7d10-4e5e-ac1d-101689ab4c78)   |
| Cámara Raspberry Pi v2 - 8MP        | Cámara HD con sensor de imagen IMX219PQ CMOS de Sony   |    ![camara-raspberry-pi-v2-8-megapixels](https://github.com/user-attachments/assets/dc3eddb9-0381-4155-a739-9be4fa2fd87a)     |
| Cable para cámara Raspberry Pi - 30cm | Extensión de la conexión entre la Raspberry Pi y la cámara, compatible con cámaras estándar y NoIR |  ![cable-para-camara-raspberry-pi-30cm](https://github.com/user-attachments/assets/4ec78ccd-0321-451b-a2c7-2062382529a4)   |
| Fuente de alimentacion Mean Well RS-50-5 (5V / 10A) | Fuente de alimentación conmutada de entrada universal 220V AC, y salida de tensión 5V y corriente 10A. Tiene una potencia nominal de 50W. Destinada a alimentar los servos |    ![fuente-de-alimentación-mean-well-conmutada-5v10a](https://github.com/user-attachments/assets/69569f52-02e3-4b27-b202-7d0fb251fa1b)    |
| Fuente de alimentación Raspberry Pi 4 - USB-C (5.1V / 3A) | Fuente de alimentación universal oficial para Raspberry Pi con cable y conector USB-C integrado, valida para todas las versiones de Raspberry pi 4 Modelo B |    ![fuente-alimentacion-raspberry-pi-4-usb-c-51v-3a](https://github.com/user-attachments/assets/7d14bb77-0550-40bf-98ee-ca546e60dddf)    |
| Controlador PWM 16 canales I2C (PCA9685) | Puede controlar hasta 16 servos con entrada de alimentación externa |      ![controlador-pwm-16-canales-i2c-pca9685](https://github.com/user-attachments/assets/3dc3e1ec-48c5-49eb-a7ec-d796e52ff795)  |
| 3 x Micro servo miniatura SG90 | Pequeño servo de plástico con fuerza de 1.8 kg/cm y ángulo de rotación de 180 grados. Alimentado por 5V |    ![micro-servo-miniatura-sg90](https://github.com/user-attachments/assets/28737a79-1ca5-47b0-9c9a-079e2833b5da)    |
| 3 x Servomotor digital MG996R | Servomotor con gran fuerza de 12kg/cm y giro de 180 grados. Voltaje ideal de 6V |     ![servomotor-digital-mg996r](https://github.com/user-attachments/assets/0c83b7ec-8ad5-4244-8814-e84b2656a039)   |

## Arquitectura software

![ModulsSoftware drawio](https://github.com/user-attachments/assets/e103d8e5-05aa-42df-9202-fe28135a4068)

### Descripción de los módulos

Puedes acceder a una descripción más detallada de cada módulo en sus directorios:

- [Image Processing](ImageProcessingModule/README.md): Módulo destinado a la detección de los bloques mediante el modelo de Ultralytics YOLO. Dentro de este módulo se incluye también el preprocesamiento de las imágenes captadas por la cámara, con la finalidad de normalizarlas y obtener un mejor reconocimiento, y la detección de su posición 3D mediante un algoritmo de Depth Estimation.
- [Movement](MovementModule/README.md): Módulo destinado al cálculo de la trayectoria del robot sin colisiones y al algoritmo de apilamiento, el cual debe tener en cuenta la estabilidad de la torre en tiempo real.
- [User Interface](UserInterfaceModule/README.md): Aplicación que permita visualizar la retransmisión de la cámara en tiempo real, iniciar y parar el funcionamiento del robot y un modo manual en el que puedas elegir qué pieza quieres que agarre el brazo robótico. 

## Bibliografía 

- Ding, Lijun, and Ardeshir Goshtasby. "On the Canny edge detector." Pattern recognition 34.3 (2001): 721-725. 

- Redmon, J., et al. "You Only Look Once: Unified, Real-Time Object Detection." CVPR, 2016. 

- Reinhard, Erik, et al. "Color transfer between images." IEEE Computer graphics and applications 21.5 (2001): 34-41. 

- Eigen, D., et al. "Depth Map Prediction from a Single Image using a Multi-Scale Deep Network." NIPS, 2014. 

- Carlos García Calvo. "Robot Braccio 5 eixos." Robòtica, Llenguatge i Planificació, 2025.

- Birkl, Reiner, Diana Wofk, and Matthias Müller. "Midas v3. 1--a model zoo for robust monocular relative depth estimation." arXiv preprint arXiv:2307.14460 (2023). 

- Rooban, S., et al. "Simulation of pick and place robotic arm using coppeliasim." 2022 6th International Conference on Computing Methodologies and Communication (ICCMC). IEEE, 2022. 

- Duda, Richard O., and Peter E. Hart. "Use of the Hough transformation to detect lines and curves in pictures." Communications of the ACM 15.1 (1972): 11-15. 

## Autores

- [Joel Montes de Oca Martínez](https://github.com/joelmdom)
- [Lucía Torrescusa Rubio](https://github.com/luciat3)
- [Sergi Díaz López](https://github.com/sergidiazlopez)
- [Arnau Giró Moliner](https://github.com/arnaugirom)

## Licencia

MIT License

Copyright (c) 2025 CatchIt!
