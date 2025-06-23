# -*- coding: utf-8 -*-
"""
Autors: 
    - Lucía Torrescusa Rubio (1633302)
    - Joel Montes de Oca Martinez (1667517)
"""

"""
# -------------------------------------------------
# IDEA INICIAL
# -------------------------------------------------

import cv2
import numpy as np

# Configuraciones iniciales
depth_threshold = 30  # Profundidad máxima para considerar como superficie plana

# Función para marcar superficies planas
def mark_flat_surfaces(image_path):
    # Leer la imagen en escala de grises para análisis de profundidad
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Normalizar para asegurar que todos los valores estén entre 0 y 255
    depth_normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    kernelOpen = np.ones((5, 1), np.uint8)
    mask = cv2.morphologyEx(depth_normalized, cv2.MORPH_OPEN, kernelOpen, iterations= 1)
    
    kernelClose = np.ones((2, 8), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernelClose, iterations=1)
    
    # Aplicar detección de bordes para resaltar superficies planas
    edges = cv2.Canny(mask, 50, 150)
    
    # Visualizar los bordes detectados
    cv2.namedWindow('Bordes Canny', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Bordes Canny', 640, 480)
    cv2.imshow('Bordes Canny', edges)
    
    # Encontrar contornos de las áreas planas
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = cv2.cvtColor(depth_normalized, cv2.COLOR_GRAY2BGR)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Ignorar ruido pequeño
            cv2.drawContours(output, [contour], -1, (0, 255, 0), 2)
            
    return output

# Ruta de la imagen de entrada (cambiar a la imagen que quieras analizar)
image_path = './1/recorte.jpg'
#image_path = "../imageTests/1.jpg"

# Marcar las superficies planas en la imagen
marked_image = mark_flat_surfaces(image_path)

# Mostrar el resultado con ventana más pequeña
cv2.namedWindow('Superficies Planas', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Superficies Planas', 640, 480)
cv2.imshow('Superficies Planas', marked_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# -------------------------------------------------

"""

"""

# -------------------------------------------------
# IDEA BASE
# -------------------------------------------------

import cv2
import numpy as np

def detectar_pieza_jenga_debug(imagen_path, mostrar=True):
    # 1. Cargar imagen
    img = cv2.imread(imagen_path)
    if img is None:
        print(f"No se pudo cargar la imagen desde {imagen_path}")
        return None

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

    # Mostrar pasos intermedios
    if mostrar:
        cv2.imshow("Original", img)
        cv2.imshow("Escala de grises", img_gray)
        cv2.imshow("Desenfocada", img_blur)

    # 2. Detección de bordes
    edges = cv2.Canny(img_blur, 100, 245)
    if mostrar:
        cv2.imshow("Bordes (Canny)", edges)

    # 3. Encontrar contornos
    contornos, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_todos_contornos = img.copy()
    cv2.drawContours(img_todos_contornos, contornos, -1, (255, 0, 0), 1)  # todos en azul

    if mostrar:
        cv2.imshow("Todos los contornos", img_todos_contornos)

    mejor_contorno = None
    mejor_approx = None

    for cnt in contornos:
        epsilon = 0.2 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # Mostrar cada contorno aproximado de 4 lados (debug)
        if len(approx) == 4 and cv2.contourArea(approx) > 1000:
            print(f"Contorno de 4 lados con área {cv2.contourArea(approx)}")
            mejor_contorno = cnt
            mejor_approx = approx
            break

    if mejor_approx is None:
        print("No se detectó ningún contorno cuadrilátero adecuado.")
        if mostrar:
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return None

    # 4. Dibujar el mejor contorno
    img_contorno = img.copy()
    cv2.drawContours(img_contorno, [mejor_approx], -1, (0, 255, 0), 2)

    puntos = mejor_approx.reshape(-1, 2)
    for i, punto in enumerate(puntos):
        cv2.circle(img_contorno, tuple(punto), 5, (0, 0, 255), -1)
        cv2.putText(img_contorno, f"P{i}", tuple(punto + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    if mostrar:
        cv2.imshow("Contorno detectado (mejor)", img_contorno)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return puntos

# Uso
puntos = detectar_pieza_jenga_debug("./1/recorte.jpg")
print("Puntos detectados:", puntos)

# -------------------------------------------------

"""

"""
# -------------------------------------------------
# ---------- DETECTA RECTA -----------
# -------------------------------------------------

import cv2
import numpy as np
import math

def angulo_entre_vectores(v1, v2):
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    cos_theta = dot / (norm1 * norm2 + 1e-8)
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    return np.degrees(angle)

def detectar_lineas_principales(imagen_path, mostrar=True, umbral_angulo=20):
    img = cv2.imread(imagen_path)
    if img is None:
        print("No se pudo cargar la imagen.")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # Detección de líneas
    lineas = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=50, maxLineGap=10)

    if lineas is None:
        print("No se detectaron líneas.")
        return None

    segmentos_validos = []
    puntos_resultantes = []

    img_lineas = img.copy()

    for i in range(len(lineas)):
        for j in range(i + 1, len(lineas)):
            x1, y1, x2, y2 = lineas[i][0]
            u1, v1, u2, v2 = lineas[j][0]

            vec1 = np.array([x2 - x1, y2 - y1])
            vec2 = np.array([u2 - u1, v2 - v1])
            angulo = angulo_entre_vectores(vec1, vec2)

            # Si el ángulo es similar y los puntos están cerca → mismos bordes
            if angulo < umbral_angulo:
                # Guardamos los extremos
                dist = np.linalg.norm(np.array([x1, y1]) - np.array([u1, v1]))
                if dist < 50:  # cerca
                    segmentos_validos.append(((x1, y1), (x2, y2)))
                    puntos_resultantes.append((x1, y1))
                    puntos_resultantes.append((x2, y2))
                    break  # no compararlo más

    # Si no hay conexiones, guardar simplemente las líneas más largas
    if not puntos_resultantes:
        longitudes = []
        for linea in lineas:
            x1, y1, x2, y2 = linea[0]
            longitudes.append(((x1, y1), (x2, y2), np.linalg.norm([x2 - x1, y2 - y1])))
        longitudes.sort(key=lambda x: -x[2])
        for i in range(min(4, len(longitudes))):
            (x1, y1), (x2, y2), _ = longitudes[i]
            segmentos_validos.append(((x1, y1), (x2, y2)))
            puntos_resultantes.append((x1, y1))
            puntos_resultantes.append((x2, y2))

    # Visualizar líneas
    for (x1, y1), (x2, y2) in segmentos_validos:
        cv2.line(img_lineas, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(img_lineas, (x1, y1), 4, (0, 0, 255), -1)
        cv2.circle(img_lineas, (x2, y2), 4, (255, 0, 0), -1)

    if mostrar:
        cv2.imshow("Lineas detectadas", img_lineas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return puntos_resultantes

# Uso
puntos = detectar_lineas_principales("./1/recorte.jpg")
print("Puntos extraídos para estimar plano:", puntos)

# -------------------------------------------------

"""

"""

# -------------------------------------------------
# --------- DETECTA RECTA 2 -------------
# -------------------------------------------------

import cv2
import numpy as np
import math

def detectar_lineas_largas_orientadas(imagen_path, mostrar=True):
    # 1. Cargar imagen y preprocesar
    img = cv2.imread(imagen_path)
    if img is None:
        print("No se pudo cargar la imagen.")
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # 2. Detectar líneas con Hough
    lineas = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50,
                             minLineLength=50, maxLineGap=10)

    if lineas is None:
        print("No se detectaron líneas.")
        return None

    lineas = lineas[:, 0, :]  # quitar una dimensión
    print(f"Líneas detectadas: {len(lineas)}")

    # 3. Calcular longitud y ángulo de cada línea
    def longitud(x1, y1, x2, y2):
        return np.hypot(x2 - x1, y2 - y1)

    def angulo(x1, y1, x2, y2):
        return math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180  # ángulo en [0,180)

    # Filtrar líneas más largas y separadas por orientación
    lineas_filtradas = []
    umbral_orientacion = 15  # grados
    max_lineas = 4

    for x1, y1, x2, y2 in sorted(lineas, key=lambda l: -longitud(*l)):
        ang = angulo(x1, y1, x2, y2)

        # Verificar si ya hay una línea con orientación similar
        if all(abs(ang - angulo(*l)) > umbral_orientacion for l in lineas_filtradas):
            lineas_filtradas.append((x1, y1, x2, y2))
            if len(lineas_filtradas) >= max_lineas:
                break

    # 4. Dibujar y guardar puntos extremos
    img_lineas = img.copy()
    puntos_resultado = []

    for i, (x1, y1, x2, y2) in enumerate(lineas_filtradas):
        cv2.line(img_lineas, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(img_lineas, (x1, y1), 5, (0, 0, 255), -1)
        cv2.circle(img_lineas, (x2, y2), 5, (255, 0, 0), -1)
        puntos_resultado.append(((x1, y1), (x2, y2)))

    if mostrar:
        cv2.imshow("Líneas detectadas", img_lineas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return puntos_resultado

# Uso del código
puntos = detectar_lineas_largas_orientadas("./1/recorte.jpg")
print("Puntos extremos de líneas seleccionadas:", puntos)

# -------------------------------------------------

"""

"""

# ------------------------------------------------- 
# PUNTOS REALTIVOS AL RECORTE SOLO
# -------------------------------------------------

import cv2
import numpy as np
import math

def detectar_lineas_largas_con_intersecciones(imagen_path, mostrar=True):
    def longitud(x1, y1, x2, y2):
        return np.hypot(x2 - x1, y2 - y1)

    def angulo(x1, y1, x2, y2):
        return math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180

    def interseccion_lineas(p1, p2, q1, q2):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = q1
        x4, y4 = q2
        denom = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
        if abs(denom) < 1e-10:
            return None
        px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
        py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
        return (int(px), int(py))

    img = cv2.imread(imagen_path)
    if img is None:
        print("No se pudo cargar la imagen.")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    lineas = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50,
                             minLineLength=50, maxLineGap=10)

    if lineas is None:
        print("No se detectaron líneas.")
        return None

    lineas = lineas[:, 0, :]
    print(f"Líneas detectadas: {len(lineas)}")

    # Paso 3: filtrar líneas más largas y con orientación distinta
    lineas_filtradas = []
    umbral_orientacion = 15
    max_lineas = 4

    for x1, y1, x2, y2 in sorted(lineas, key=lambda l: -longitud(*l)):
        ang = angulo(x1, y1, x2, y2)
        if all(abs(ang - angulo(*l)) > umbral_orientacion for l in lineas_filtradas):
            lineas_filtradas.append((x1, y1, x2, y2))
            if len(lineas_filtradas) >= max_lineas:
                break

    # Paso 4: construir conjunto de puntos sin repetir extremos cercanos
    extremos = []
    intersecciones = []

    # Comparar pares de líneas para detectar extremos cercanos
    usados = set()

    for i in range(len(lineas_filtradas)):
        xi1, yi1, xi2, yi2 = lineas_filtradas[i]
        for j in range(i + 1, len(lineas_filtradas)):
            xj1, yj1, xj2, yj2 = lineas_filtradas[j]

            extremos_i = [(xi1, yi1), (xi2, yi2)]
            extremos_j = [(xj1, yj1), (xj2, yj2)]

            for pi in extremos_i:
                for pj in extremos_j:
                    if np.linalg.norm(np.array(pi) - np.array(pj)) < 30:
                        inter = interseccion_lineas((xi1, yi1), (xi2, yi2),
                                                    (xj1, yj1), (xj2, yj2))
                        if inter is not None:
                            intersecciones.append(inter)
                            usados.add(pi)
                            usados.add(pj)

    # Agregar extremos no usados
    for x1, y1, x2, y2 in lineas_filtradas:
        for punto in [(x1, y1), (x2, y2)]:
            if punto not in usados:
                extremos.append(punto)

    puntos_finales = intersecciones + extremos

    # Visualización
    img_lineas = img.copy()
    for (x1, y1, x2, y2) in lineas_filtradas:
        cv2.line(img_lineas, (x1, y1), (x2, y2), (0, 255, 0), 2)

    for i, pt in enumerate(puntos_finales):
        cv2.circle(img_lineas, pt, 6, (0, 0, 255), -1)
        cv2.putText(img_lineas, f"P{i}", (pt[0] + 5, pt[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    if mostrar:
        cv2.imshow("Líneas e intersecciones", img_lineas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return puntos_finales

# Uso
puntos = detectar_lineas_largas_con_intersecciones("./1/recorte.jpg")
print("Puntos clave para plano superior:", puntos)

# -------------------------------------------------

"""



import cv2
import numpy as np
import math
import os

def detectar_lineas_largas_con_intersecciones(imagen_recorte_path, imagen_original_path, carpeta_salida, mostrar=True, th_l=50, th_u=70):
    def longitud(x1, y1, x2, y2):
        return np.hypot(x2 - x1, y2 - y1)

    def angulo(x1, y1, x2, y2):
        return math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180

    def interseccion_lineas(p1, p2, q1, q2):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = q1
        x4, y4 = q2
        denom = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
        if abs(denom) < 1e-10:
            return None
        px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
        py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
        return (int(px), int(py))

    # Leer bbox_coords para obtener offsets
    ruta_bbox = os.path.join(carpeta_salida, "bbox_coords.npy")
    if not os.path.exists(ruta_bbox):
        print(f"❌ No se encontró el archivo bbox_coords.npy en {carpeta_salida}")
        return None
    bbox_coords = np.load(ruta_bbox)
    x_offset, y_offset = int(bbox_coords[0]), int(bbox_coords[1])

    img_recorte = cv2.imread(imagen_recorte_path)
    
    if img_recorte is None:
        print("No se pudo cargar la imagen recortada.")
        return None

    gray = cv2.cvtColor(img_recorte, cv2.COLOR_BGR2GRAY)
    
    if mostrar:
        cv2.imshow("gray", gray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    if mostrar:
        cv2.imshow("blur", blur)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    """
    _, thresh1 = cv2.threshold(blur,145,255,cv2.THRESH_BINARY)
    
    cv2.imshow("threshold", thresh1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # - Calculo Automatico Thresholds
    median = np.median(thresh1)
    sigma = 0.9
    t_lower = int(max(0, (1.0 - sigma) * median))
    t_upper = int(min(255, (1.0 + sigma) * median))
    
    edges = cv2.Canny(thresh1, t_lower, t_upper)
    """
    
    # AJUSTAR A ILUMINACIÓN
    #edges = cv2.Canny(blur, 50, 150)
    edges = cv2.Canny(blur, th_l, th_u)
    
    if mostrar:
        cv2.imshow("edges", edges)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    lineas = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50,
                             minLineLength=50, maxLineGap=10)

    if lineas is None:
        print("No se detectaron líneas.")
        return None

    lineas = lineas[:, 0, :]
    print(f"Líneas detectadas: {len(lineas)}")

    # Filtrar líneas largas y orientaciones diferentes
    lineas_filtradas = []
    umbral_orientacion = 15
    max_lineas = 4

    for x1, y1, x2, y2 in sorted(lineas, key=lambda l: -longitud(*l)):
        ang = angulo(x1, y1, x2, y2)
        if all(abs(ang - angulo(*l)) > umbral_orientacion for l in lineas_filtradas):
            lineas_filtradas.append((x1, y1, x2, y2))
            if len(lineas_filtradas) >= max_lineas:
                break

    extremos = []
    intersecciones = []
    usados = set()

    for i in range(len(lineas_filtradas)):
        xi1, yi1, xi2, yi2 = lineas_filtradas[i]
        for j in range(i + 1, len(lineas_filtradas)):
            xj1, yj1, xj2, yj2 = lineas_filtradas[j]

            extremos_i = [(xi1, yi1), (xi2, yi2)]
            extremos_j = [(xj1, yj1), (xj2, yj2)]

            for pi in extremos_i:
                for pj in extremos_j:
                    if np.linalg.norm(np.array(pi) - np.array(pj)) < 30:
                        inter = interseccion_lineas((xi1, yi1), (xi2, yi2),
                                                    (xj1, yj1), (xj2, yj2))
                        if inter is not None:
                            intersecciones.append(inter)
                            usados.add(pi)
                            usados.add(pj)

    for x1, y1, x2, y2 in lineas_filtradas:
        for punto in [(x1, y1), (x2, y2)]:
            if punto not in usados:
                extremos.append(punto)

    puntos_finales = intersecciones + extremos

    # Visualizar puntos en recorte (coordenadas relativas)
    if mostrar:
        img_puntos_recorte = img_recorte.copy()
        for pt in puntos_finales:
            cv2.circle(img_puntos_recorte, pt, 6, (0, 255, 0), -1)
        cv2.imshow("Puntos en recorte (relativo)", img_puntos_recorte)

    # Ajustar puntos a coordenadas globales sumando offsets bbox
    puntos_globales = [(int(x + x_offset), int(y + y_offset)) for (x, y) in puntos_finales]

    # Visualizar puntos en imagen original (coordenadas globales)
    if mostrar:
        img_original = cv2.imread(imagen_original_path)
        if img_original is None:
            print("No se pudo cargar la imagen original, mostrando recorte con puntos globales.")
            img_original = img_recorte.copy()
        for pt in puntos_globales:
            cv2.circle(img_original, pt, 6, (0, 0, 255), -1)
        cv2.imshow("Puntos en imagen original (global)", img_original)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Guardar puntos globales
    ruta_puntos = os.path.join(carpeta_salida, "pts_plano_pieza.npy")
    np.save(ruta_puntos, np.array(puntos_globales, dtype=np.int32))
    print(f"✅ {len(puntos_globales)} puntos guardados en '{ruta_puntos}':")
    for i, pt in enumerate(puntos_globales):
        print(f"  Punto {i+1}: {pt}")

    return puntos_globales

def main():
    
    carpeta_salida = "./arxius_necessaris"   # Carpeta donde están bbox_coords.npy y donde se guardarán puntos
    imagen_recorte_path = "./arxius_necessaris/recorte.jpg"   # Ruta a la imagen recortada (input)
    imagen_original_path = "../test/p1.jpg" # Ruta a la imagen original
    
    puntos = detectar_lineas_largas_con_intersecciones(imagen_recorte_path, imagen_original_path, carpeta_salida, mostrar=True)
    

if __name__ == "__main__":
    main()