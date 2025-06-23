En el módulo de procesamiento de imagen, se encuentran los siguientes ficheros:

## Generación de dataset
- **PecesJenga.ttt**: Entorno de CoppeliaSim Scene creado para generar el dataset de piezas en diferentes posiciones.
- **PecesGeneracio.py**: Código de Python conectado a la escena de Coppelia para aleatoriamente generar diferentes imágenes con las piezas colocadas de diferentes maneras.

## Dataset
- **images/**: Imágenes generadas sintéticamente.
- **labels/**: Etiquetado manual de las imágenes.

## Entrenamiento de modelo YOLO
- **data.yaml**: Configuración de elementos (enrutamiento y etiquetas).
- **trainingYOLO.py**: Ejecutar este código activará el entrenamiento.
- **train9/**: Mejor modelo entrenado.

## Detección de posiciones

**execute_project.py**: Ejecuta la detección de posición, importando los siguientes ficheros:

- **selectCorners.py**: Seleccionar los tres puntos del marco.
- **depthMap.py**: Ejecuta MiDaS para generar el mapa de profundidad.
- **LUT_real_to_simulated.py**: Preprocesa las imágenes mediante transformada de Reinhard para asemejarlas a las imágenes simuladas.
- **getCoordinates.py**: Devuelve coordenadas reales a partir de la imagen.
- **pieza_individual.py**: Detecta el plano de la cara de la pieza.
- **gradosPieza.py**: Calcula intersección entre los planos del fondo y de la pieza para devolver la orientación.

## Útiles
- **yolov8n.pt**
- **sim.py**
- **simConst.py**
- **remoteApi.dll**

***

Gracias a esto, pasamos de una escena sintética:

![nCaotico_003_zenital](https://github.com/user-attachments/assets/c1cbef52-c1a5-43e3-bc7d-5f44e0726a3e)

A una imagen real con detección mediante YOLO:

![image](https://github.com/user-attachments/assets/84c5aa96-c6b5-424f-8621-bda54f4f6bcc)

A, finalmente, una matriz de posiciones y orientaciones detectadas. El proceso final termina siendo así:

![DemoVC](https://github.com/user-attachments/assets/a2b6ef13-378e-4f7f-bd40-97c9095b6ff8)
