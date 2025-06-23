# -*- coding: utf-8 -*-
"""
Autors: 
    - Lucía Torrescusa Rubio (1633302)
    - Joel Montes de Oca Martinez (1667517)
"""

import torch
from ultralytics import YOLO
import numpy as np
import cv2
import os

def coordenadas(modelo_path, imagen_path, carpeta_salida):
    
    # === Crear carpeta si no existe ===
    os.makedirs(carpeta_salida, exist_ok=True)
    
    # === Cargar modelo YOLO ===
    modelo = YOLO(modelo_path)
    
    # === Ejecutar inferencia ===
    resultado = modelo(imagen_path)[0]
    
    # === Leer imagen original
    img = cv2.imread(imagen_path)
    img_biggest = img.copy()  # para dibujar solo la mayor bbox
    
    w_img, h_img = img.shape[1], img.shape[0]
    
    # === Inicializar selección de bbox más grande
    bbox_mayor = None
    mayor_area = 0
    
    zona_ignorada = {
        "x_min": 1450,
        "x_max": 1600,
        "y_min": 900,
        "y_max": 1080
    }
    
    # === Procesar todas las detecciones
    for box in resultado.boxes.data:
        x1, y1, x2, y2, conf, cls = box.cpu().numpy()
        area = (x2 - x1) * (y2 - y1)
        bbox_center_x = (x1 + x2) / 2
        bbox_center_y = (y1 + y2) / 2
    
        # --- Filtros personalizados ---
        if area > 30000:
            continue
        if area < 10000:
            continue
        if bbox_center_x > 0.9 * w_img or bbox_center_y > 0.9 * h_img:
            continue
    
        """
        # AÑADIR ZONA MONTAJE TORRE A IGNORAR
        if (zona_ignorada["x_min"] <= bbox_center_x <= zona_ignorada["x_max"] and
            zona_ignorada["y_min"] <= bbox_center_y <= zona_ignorada["y_max"]):
            continue
        """
        # Dibujar detección válida (verde)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(img, f"{conf:.2f}", (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    
        # Seleccionar bbox de mayor área
        if area > mayor_area:
            mayor_area = area
            bbox_mayor = (x1, y1, x2, y2)
    
    # === Dibujar la bbox más grande (rojo) en img_biggest
    if bbox_mayor:
        x1, y1, x2, y2 = map(int, bbox_mayor)
        cv2.rectangle(img_biggest, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img_biggest, "MAYOR AREA", (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        print(f"Bounding box más grande: área = {mayor_area:.0f}, coords = ({x1}, {y1}) - ({x2}, {y2})")
    else:
        print("No se encontró ninguna bbox válida.")
    
    
    
    # === Aplicar MiDaS para obtener profundidad ===
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas.to(device).eval()
    
    # Preprocesar imagen
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_tensor = midas_transforms(img_rgb).to(device)
    
    with torch.no_grad():
        prediction = midas(input_tensor)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze()
        depth_map = prediction.cpu().numpy()
    
    # === Centro de la bbox más grande
    if bbox_mayor:
        x1, y1, x2, y2 = bbox_mayor
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
    
        # Profundidad MiDaS en ese punto
        z_midas = depth_map[cy, cx]
        print(f"Profundidad MiDaS en la pieza más cercana (cx={cx}, cy={cy}): Z = {z_midas:.4f}")
    else:
        print("No se puede obtener profundidad: no se encontró bbox válida.")
    
    # === Cargar puntos de imagen previamente guardados manualmente
    ruta_pts = carpeta_salida+"/pts_plano_base.npy"
    if bbox_mayor and os.path.exists(ruta_pts):
        pts_img = np.load(ruta_pts).astype(np.float32)
    
        # Coordenadas reales correspondientes (en centímetros)
        pts_real = np.array([
            [0, 0],     # esquina inferior izquierda (cercana a cámara)
            [0, 40],   # esquina superior derecha
            [40, 40],    # esquina superior izquierda
        ], dtype=np.float32)
    
        if pts_img.shape == (3, 2):
            # Calcular transformación afín (3 puntos)
            M = cv2.getAffineTransform(pts_img, pts_real)
    
            # Aplicar a (cx, cy)
            punto_h = np.array([cx, cy, 1])
            x_real, y_real = np.dot(M, punto_h)
            
            # === Obtener Z relativa al plano base (tabla)
            # Usamos la primera esquina (0,0) del marco como referencia
            x_base, y_base = int(pts_img[0][0]), int(pts_img[0][1])  # esquina inferior izquierda en píxeles
            z_midas_base = depth_map[y_base, x_base]
            z_midas_objeto = depth_map[cy, cx]
            
            # Diferencia de profundidad (en escala MiDaS)
            z_midas_diff = z_midas_base - z_midas_objeto  # positiva si el objeto está por encima del plano
            
            # === Opcional: factor para convertir a centímetros reales (calibrado previamente)
            # Ejemplo: una pieza de 3.5 cm de alto da diferencia z_midas_diff ≈ 55 → factor ≈ 0.0636
            factor_z = 0.002
            z_real = z_midas_diff * factor_z  # altura sobre la tabla en cm
    
    
    
            print("Coordenadas reales estimadas de la pieza:")
            print("X = ", x_real,"cm, Y = ", y_real, "cm, Z = ", z_real, "cm")
        else:
            print("❌ El archivo de puntos no contiene exactamente 3 puntos.")
    else:
        print("❌ No se pudo cargar la homografía: falta bbox válida o archivo de puntos.")
    
    # === Guardar ambas imágenes
    nombre_imagen = os.path.basename(imagen_path)
    ruta_yolo = os.path.join(carpeta_salida, nombre_imagen)
    ruta_biggest = os.path.join(carpeta_salida, "biggestboundingbox.jpg")
    
    cv2.imwrite(ruta_yolo, img)
    cv2.imwrite(ruta_biggest, img_biggest)
    
    # === Dibujar coordenadas reales sobre imagen de mayor área
    img_coords = img_biggest.copy()
    
    if bbox_mayor and x_real is not None and y_real is not None and z_real is not None:
        # Coordenadas de píxel del centro de la bbox mayor
        punto_px = (cx, cy)
    
        # Dibujar el punto
        cv2.circle(img_coords, punto_px, 8, (255, 0, 255), -1)
    
        # Escribir coordenadas reales
        texto = f"X={x_real:.1f}cm, Y={y_real:.1f}cm, Z={z_real:.1f}cm"
        cv2.putText(img_coords, texto, (punto_px[0] + 15, punto_px[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        # Guardar imagen final
        ruta_coords = os.path.join(carpeta_salida, "coordenadas_reales.jpg")
        cv2.imwrite(ruta_coords, img_coords)
        print(f"🖼 Imagen con coordenadas guardada en: {ruta_coords}")
    else:
        print("⚠️ No se pudo dibujar coordenadas: faltan datos.")
        
        
        
        
    # === Recortar y guardar imagen de la bbox más grande (al final del proceso)
    if bbox_mayor:
        img = cv2.imread(imagen_path)
        x1, y1, x2, y2 = map(int, bbox_mayor)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w_img, x2)
        y2 = min(h_img, y2)
        img_recorte = img[y1:y2, x1:x2]
    
        # Guardar imagen recortada con nombre personalizado
        nombre_recorte = "recorte.jpg"
        ruta_recorte = os.path.join(carpeta_salida, nombre_recorte)
        cv2.imwrite(ruta_recorte, img_recorte)
        print(f"📦 Imagen recortada FINAL guardada en: {ruta_recorte}")
        print(f"📐 Coordenadas de la bbox en la imagen original: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
       
       # Guardar coordenadas bbox para referencias futuras (ej: mapear puntos del recorte al original)
        bbox_coords = np.array([x1, y1, x2, y2])
        np.save(os.path.join(carpeta_salida, "bbox_coords.npy"), bbox_coords)
    else:
        print("⚠️ No se pudo recortar (final): no hay bbox válida.")
    
    
def main():
    # === Configura rutas ===
    modelo_path = "../yolo/runs/detect/train/weights/best.pt"
    imagen_path = "../test/img_tratada.jpg"
    carpeta_salida = "arxius_necessaris"
    
    coordenadas(modelo_path, imagen_path, carpeta_salida)


if __name__ == "__main__":
    main()