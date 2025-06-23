# -*- coding: utf-8 -*-
"""
Autors: 
    - Lucía Torrescusa Rubio (1633302)
    - Joel Montes de Oca Martinez (1667517)
"""

import cv2
import numpy as np
import os

def seleccionarBordes(imagen_path, output_path):
    
    # === Crear carpeta si no existe ===
    os.makedirs(output_path, exist_ok=True)
    
    # Leer imagen original
    img = cv2.imread(imagen_path)
    assert img is not None, "No se pudo cargar la imagen."
    
    # Crear una copia redimensionada para visualizar (sin afectar las coordenadas)
    scale = 0.5  # puedes ajustar (0.5 = 50% del tamaño original)
    img_display = cv2.resize(img.copy(), (0, 0), fx=scale, fy=scale)
    
    
    clicks = []
    
    # Función de clic
    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 3:
            # Convertir coordenadas desde imagen reducida a original
            x_orig = int(x / scale)
            y_orig = int(y / scale)
            clicks.append((x_orig, y_orig))
    
            print(f"Punto {len(clicks)}: ({x_orig}, {y_orig})")
            cv2.circle(img_display, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Selecciona (5 puntos) esquinas del marco", img_display)
    
            if len(clicks) == 3:
                np.save(output_path+"/pts_plano_base.npy", np.array(clicks, dtype=np.float32))
                print(f"\n✅ Guardado como: {output_path}/pts_plano_base.npy")
                print(f"Coordenadas guardadas: {clicks}")
                cv2.destroyAllWindows()
    
    # Mostrar ventana reducida
    cv2.imshow("Selecciona (5 puntos) esquinas del marco", img_display)
    cv2.setMouseCallback("Selecciona (5 puntos) esquinas del marco", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def main():
    fondo_path = "../test/fondo.jpg"
    out_path = "./arxius_necessaris"

    seleccionarBordes(fondo_path, out_path)
    
if __name__ == "__main__":
    main()






"""
# --------------------------------------------------------
# --- AUTOMATICO - CLICK SOBRE LINEAS PARA SELECCIONAR ---
# --------------------------------------------------------
# - no funciona con getCoordinates.py

import cv2
import numpy as np

imagen_path = "../test/fondo.jpg"
output_path = "./arxius_necessaris/pts_plano_base.npy"

# Leer imagen
img = cv2.imread(imagen_path)
assert img is not None, "No se pudo cargar la imagen."
original_img = img.copy()

# Escala de visualización
scale = 0.5
img_display = cv2.resize(img.copy(), (0, 0), fx=scale, fy=scale)

# Detectar bordes y líneas
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 50, 150)
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

selected_lines = []

# Dibujar todas las líneas detectadas
if lines is not None:
    all_lines = [line[0] for line in lines]
    display_lines = img_display.copy()
    for x1, y1, x2, y2 in all_lines:
        cv2.line(display_lines, (int(x1*scale), int(y1*scale)), (int(x2*scale), int(y2*scale)), (0, 255, 0), 2)
else:
    print(" No se detectaron líneas.")
    exit(1)

# Función para encontrar línea más cercana al clic
def line_distance(x, y, line):
    x1, y1, x2, y2 = line
    p = np.array([x, y])
    a = np.array([x1, y1]) * scale
    b = np.array([x2, y2]) * scale
    ab = b - a
    ap = p - a
    proj = np.dot(ap, ab) / np.linalg.norm(ab)**2
    proj_point = a + proj * ab
    dist = np.linalg.norm(proj_point - p)
    return dist

# Función de clic
def click_event(event, x, y, flags, param):
    global display_lines
    if event == cv2.EVENT_LBUTTONDOWN:
        closest_idx = None
        min_dist = float('inf')
        for i, line in enumerate(all_lines):
            dist = line_distance(x, y, line)
            if dist < 20 and dist < min_dist:
                closest_idx = i
                min_dist = dist
        if closest_idx is not None and closest_idx not in selected_lines:
            selected_lines.append(closest_idx)
            x1, y1, x2, y2 = all_lines[closest_idx]
            cv2.line(display_lines, (int(x1*scale), int(y1*scale)), (int(x2*scale), int(y2*scale)), (0, 0, 255), 3)
            print(f"✅ Línea seleccionada: ({x1}, {y1}) -> ({x2}, {y2})")

# Ventana interactiva
cv2.namedWindow("Selecciona líneas del plano")
cv2.setMouseCallback("Selecciona líneas del plano", click_event)

while True:
    cv2.imshow("Selecciona líneas del plano", display_lines)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Presiona 'q' para continuar
        break

cv2.destroyAllWindows()

# === Calcular intersecciones entre líneas seleccionadas ===
def intersection(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    denom = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
    if denom == 0:
        return None
    px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
    return int(px), int(py)

# Calcular intersecciones
intersections = []
for i in range(len(selected_lines)):
    for j in range(i + 1, len(selected_lines)):
        pt = intersection(all_lines[selected_lines[i]], all_lines[selected_lines[j]])
        if pt:
            intersections.append(pt)

# Filtrar duplicados cercanos
filtered = []
for pt in intersections:
    if all(np.linalg.norm(np.array(pt) - np.array(p)) > 20 for p in filtered):
        filtered.append(pt)

# Dibujar y guardar
for pt in filtered[:5]:
    cv2.circle(img, pt, 8, (255, 0, 0), -1)

cv2.imshow("Resultado final", cv2.resize(img, (960, 540)))
cv2.waitKey(0)
cv2.destroyAllWindows()

if len(filtered) >= 5:
    np.save(output_path, np.array(filtered[:5], dtype=np.float32))
    print(f"\n Guardado como: {output_path}")
    print(f"Coordenadas guardadas: {filtered[:5]}")
else:
    print(" No se detectaron suficientes intersecciones válidas.")
    
"""