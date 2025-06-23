# -*- coding: utf-8 -*-
"""
Created on Sun May 18 13:15:43 2025

@author: Joel
"""

import cv2
import os
from LUT import LUT_real_to_simulated
from coordinates import selectCorners 
from coordinates.depth_map import depth_map
from coordinates import getCoordinates
from coordinates import pieza_individual
from coordinates import gradosPieza

import matplotlib.pyplot as plt


#-----------------------------------------------
# PATHS NECESARIOS
#-----------------------------------------------
# RELATIVO
modelo_path = "./yolo/runs/detect/train/weights/best.pt"
fondo_path = "./test/p1.jpg"
img_input_path = "./test/p1.jpg"
img_folder = "./test/"
arxius_necessaris = "./coordinates/arxius_necessaris"
depth_map_path = "./coordinates/depth_map/depth_map.npy"

# ABSOLUTO
modelo_abs = os.path.abspath(modelo_path)
fondo_abs = os.path.abspath(fondo_path)
img_abs = os.path.abspath(img_input_path)
img_folder_abs = os.path.abspath(img_folder)
arxius_abs = os.path.abspath(arxius_necessaris)
dmap_abs = os.path.abspath(depth_map_path)



# ------------------ EJECUTAR LA PRIMERA VEZ SOLO --------------------------

#-----------------------------------------------
# SELECCIONAR PUNTOS MESA
#-----------------------------------------------
selectCorners.seleccionarBordes(fondo_abs, arxius_abs)


#-----------------------------------------------
# DEPTH MAP FONDO
#-----------------------------------------------
depth_map.generar_mapa_profundidad(fondo_abs,dmap_abs)




# ------------------ PROCESAMIENTO PIEZAS ------------------
"""
#-----------------------------------------------
# PREPROCESAMIENTO - "LUT"/TRANSFORMADA REINHARD
#-----------------------------------------------
img_input = cv2.imread(img_abs)

img_tratada = LUT_real_to_simulated.aplicar_transformacion(img_input, transform_path="./LUT/transform_reinhard.json")

plt.imshow(img_tratada[:, :, ::-1])
plt.axis('off')
plt.show()

# GUARDAR IMAGEN TRATADA EN EL DIRECTORIO BASE
tratada_abs = os.path.join(img_folder_abs, "img_tratada.jpg")
cv2.imwrite(tratada_abs,img_tratada)
"""

#-----------------------------------------------
# COORDENADAS PIEZA
#-----------------------------------------------
#getCoordinates.coordenadas(modelo_abs, tratada_abs, arxius_abs)
getCoordinates.coordenadas(modelo_abs, img_abs, arxius_abs)



#-----------------------------------------------
# DETECTAR PLANO/RECTA PIEZA
#-----------------------------------------------
recorte_path = os.path.join(arxius_abs, "recorte.jpg")

#pieza_individual.detectar_lineas_largas_con_intersecciones(recorte_path, tratada_abs, arxius_abs, True)
pieza_individual.detectar_lineas_largas_con_intersecciones(recorte_path, img_abs, arxius_abs, True, th_l=50, th_u=80)
 


#-----------------------------------------------
# GRADOS BASE-PIEZA
#-----------------------------------------------
img_coord_path = os.path.join(arxius_abs, "coordenadas_reales.jpg")
coord_bb = os.path.join(arxius_abs, "bbox_coords.npy")
img_path = os.path.join(arxius_abs, "biggestboundingbox.jpg")
pts_plano_base_path = os.path.join(arxius_abs, "pts_plano_base.npy")
pts_pieza_path = os.path.join(arxius_abs, "pts_plano_pieza.npy")

gradosPieza.ejecucion(img_coord_path, coord_bb, img_path, dmap_abs, pts_plano_base_path, pts_pieza_path)