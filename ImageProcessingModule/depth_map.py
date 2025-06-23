# -*- coding: utf-8 -*-
"""
Autors: 
    - Lucía Torrescusa Rubio (1633302)
    - Joel Montes de Oca Martinez (1667517)
"""

import torch
import cv2
import numpy as np
import os

def generar_mapa_profundidad(imagen_path, salida_path):
    # Cargar modelo MiDaS
    print("🔄 Cargando modelo MiDaS...")
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas.to(device).eval()

    # Leer y preprocesar la imagen
    print(f"📷 Procesando imagen: {imagen_path}")
    img = cv2.imread(imagen_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_tensor = midas_transforms(img_rgb).to(device)

    # Inferencia de profundidad
    with torch.no_grad():
        prediction = midas(input_tensor)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze()
        depth_map = prediction.cpu().numpy()

    # Guardar mapa de profundidad
    np.save(salida_path, depth_map)
    print(f" Mapa de profundidad guardado en: {salida_path}")



def main():
    imagen_path = "../test/fondo.jpg"
    salida_path = "./depth_map/depth_map.npy"
    generar_mapa_profundidad(imagen_path, salida_path)

if __name__ == "__main__":
    main()