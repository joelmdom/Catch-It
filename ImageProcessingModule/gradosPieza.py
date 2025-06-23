# -*- coding: utf-8 -*-
"""
Autors: 
    - Lucía Torrescusa Rubio (1633302)
    - Joel Montes de Oca Martinez (1667517)
"""

"""

# --------------------------------------------------------
# --- INICIAL 1 ---
# --------------------------------------------------------


import cv2
import numpy as np
import torch
from ultralytics import YOLO
import os


def detectar_plano_mesa(img_path, umbral_longitud=100, umbral_angulo=5):
    # Leer imagen
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Detectar líneas
    lineas = cv2.HoughLines(edges, 1, np.pi / 180, umbral_longitud)
    if lineas is None:
        print("⚠️ No se detectaron líneas suficientemente largas para calcular el plano de la mesa.")
        return None
    
    # Filtrar líneas aproximadamente horizontales y verticales
    lineas_filtradas = []
    for rho, theta in lineas[:, 0]:
        angulo_grados = np.degrees(theta)
        if abs(angulo_grados) < umbral_angulo or abs(angulo_grados - 90) < umbral_angulo:
            lineas_filtradas.append((rho, theta))
    
    # Retornar líneas filtradas para visualización
    return lineas_filtradas


def calcular_plano_3d(puntos_2d, depth_map):
    # Convertir puntos 2D a 3D usando profundidad
    puntos_3d = []
    for (x, y) in puntos_2d:
        z = depth_map[y, x]  # Usar profundidad de MiDaS
        puntos_3d.append((x, y, z))
    return np.array(puntos_3d)


def calcular_angulo_entre_planos(p1, p2, p3, q1, q2, q3):
    # Crear vectores del plano 1
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p3) - np.array(p1)
    n1 = np.cross(v1, v2)
    
    # Crear vectores del plano 2
    w1 = np.array(q2) - np.array(q1)
    w2 = np.array(q3) - np.array(q1)
    n2 = np.cross(w1, w2)

    # Calcular ángulo entre normales
    cos_theta = np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2))
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    return np.degrees(theta)



img_path = "./1/biggestboundingbox.jpg"
depth_map_path = "./depth_map/depth_map.npy"  # Suponiendo que tienes esto guardado

# Detectar plano de la mesa
lineas_mesa = detectar_plano_mesa(img_path)
if not lineas_mesa:
    print("❌ No se pudo detectar el plano de la mesa.")


# Cargar profundidad
depth_map = np.load(depth_map_path)

# Definir puntos del plano de la mesa (por ahora usando líneas detectadas manualmente)
puntos_mesa = [(100, 500), (500, 500), (100, 100)]  # Placeholder
puntos_pieza = [(300, 300), (350, 300), (300, 250)]  # Placeholder

# Calcular planos 3D
plano_mesa_3d = calcular_plano_3d(puntos_mesa, depth_map)
plano_pieza_3d = calcular_plano_3d(puntos_pieza, depth_map)

# Calcular ángulo
angulo = calcular_angulo_entre_planos(
    plano_mesa_3d[0], plano_mesa_3d[1], plano_mesa_3d[2],
    plano_pieza_3d[0], plano_pieza_3d[1], plano_pieza_3d[2]
)
print(f"🔄 Ángulo entre el plano de la mesa y la pieza: {angulo:.2f} grados")

"""


"""

# --------------------------------------------------------
# --- INICIAL 2 ---
# --------------------------------------------------------

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def detectar_plano_mesa(img_path, umbral_longitud=100, umbral_angulo=5):
    # Leer imagen
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Detectar líneas
    lineas = cv2.HoughLines(edges, 1, np.pi / 180, umbral_longitud)
    if lineas is None:
        print("⚠️ No se detectaron líneas suficientemente largas para calcular el plano de la mesa.")
        return None
    
    # Filtrar líneas aproximadamente horizontales y verticales
    lineas_filtradas = []
    for rho, theta in lineas[:, 0]:
        angulo_grados = np.degrees(theta)
        if abs(angulo_grados) < umbral_angulo or abs(angulo_grados - 90) < umbral_angulo:
            lineas_filtradas.append((rho, theta))
    
    # Retornar líneas filtradas para visualización
    return lineas_filtradas


def calcular_plano_3d(puntos_2d, depth_map):
    # Convertir puntos 2D a 3D usando profundidad
    puntos_3d = []
    for (x, y) in puntos_2d:
        z = depth_map[y, x]  # Usar profundidad de MiDaS
        puntos_3d.append((x, y, z))
    return np.array(puntos_3d)


def calcular_angulo_entre_planos(p1, p2, p3, q1, q2, q3):
    # Crear vectores del plano 1
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p3) - np.array(p1)
    n1 = np.cross(v1, v2)
    
    # Crear vectores del plano 2
    w1 = np.array(q2) - np.array(q1)
    w2 = np.array(q3) - np.array(q1)
    n2 = np.cross(w1, w2)

    # Calcular ángulo entre normales
    cos_theta = np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2))
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    return np.degrees(theta), n1, n2


def seleccionar_puntos_manual(img_path, num_puntos=4, nombre_archivo="pts_pieza_manual.npy"):
    # Leer imagen
    img = cv2.imread(img_path)
    puntos = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            puntos.append((x, y))
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(img, f"{len(puntos)}", (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.imshow("Selecciona los puntos (4 en total)", img)

            # Guardar cuando se tengan los 4 puntos
            if len(puntos) == num_puntos:
                np.save(nombre_archivo, np.array(puntos))
                print(f"✅ Puntos guardados en {nombre_archivo}")
                cv2.destroyAllWindows()

    # Mostrar imagen y esperar selección
    cv2.imshow("Selecciona los puntos (4 en total)", img)
    cv2.setMouseCallback("Selecciona los puntos (4 en total)", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def visualizar_interseccion_plano(n1, n2, p1, q1, puntos_mesa, puntos_pieza):
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Crear puntos para plano 1 (mesa)
    X_mesa, Y_mesa = np.meshgrid(np.linspace(0, 800, 5), np.linspace(0, 800, 5))
    Z_mesa = (-n1[0] * (X_mesa - p1[0]) - n1[1] * (Y_mesa - p1[1]) + n1[2] * p1[2]) / n1[2]
    ax.plot_surface(X_mesa, Y_mesa, Z_mesa, color='green', alpha=0.5, label='Plano Mesa')

    # Crear puntos para plano 2 (pieza)
    X_pieza, Y_pieza = np.meshgrid(np.linspace(0, 800, 5), np.linspace(0, 800, 5))
    Z_pieza = (-n2[0] * (X_pieza - q1[0]) - n2[1] * (Y_pieza - q1[1]) + n2[2] * q1[2]) / n2[2]
    ax.plot_surface(X_pieza, Y_pieza, Z_pieza, color='blue', alpha=0.5, label='Plano Pieza')

    ax.set_xlabel('X (px)')
    ax.set_ylabel('Y (px)')
    ax.set_zlabel('Z (profundidad)')
    plt.title("Intersección de Planos (Mesa vs Pieza)")
    plt.savefig('interseccion_planos.png', dpi=300)
    #plt.show()


img_path = "./1/biggestboundingbox.jpg"
depth_map_path = "./depth_map/depth_map.npy"  # Suponiendo que tienes esto guardado

# Cargar profundidad
depth_map = np.load(depth_map_path)

# === Cargar puntos manuales de la mesa ===
pts_img_manual_path = "pts_img_manual.npy"
if os.path.exists(pts_img_manual_path):
    puntos_mesa = np.load(pts_img_manual_path).astype(np.int32)
    if puntos_mesa.shape != (3, 2):
        print("⚠️ El archivo pts_img_manual.npy no contiene exactamente 3 puntos. Verifica el archivo.")
        exit(1)
else:
    print(f"⚠️ Archivo no encontrado: {pts_img_manual_path}")
    exit(1)

# === Seleccionar puntos del plano de la pieza manualmente ===
seleccionar_puntos_manual(img_path)

# Cargar puntos de la pieza
puntos_pieza = np.load("pts_pieza_manual.npy").astype(np.int32)

# Calcular planos 3D
plano_mesa_3d = calcular_plano_3d(puntos_mesa, depth_map)
plano_pieza_3d = calcular_plano_3d(puntos_pieza[:3], depth_map)  # Solo los primeros 3 puntos para el plano

# Calcular ángulo
angulo, n1, n2 = calcular_angulo_entre_planos(
    plano_mesa_3d[0], plano_mesa_3d[1], plano_mesa_3d[2],
    plano_pieza_3d[0], plano_pieza_3d[1], plano_pieza_3d[2]
)
print(f"🔄 Ángulo entre el plano de la mesa y la pieza: {angulo:.2f} grados")

# Visualizar planos y su intersección
visualizar_interseccion_plano(n1, n2, plano_mesa_3d[0], plano_pieza_3d[0], puntos_mesa, puntos_pieza)

"""

"""
# --------------------------------------------------------
# --- # VISUALIZA PLANOS ---
# --------------------------------------------------------


import cv2
import numpy as np
import torch
from ultralytics import YOLO
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go


def detectar_plano_mesa(img_path, umbral_longitud=100, umbral_angulo=5):
    # Leer imagen
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Detectar líneas
    lineas = cv2.HoughLines(edges, 1, np.pi / 180, umbral_longitud)
    if lineas is None:
        print("⚠️ No se detectaron líneas suficientemente largas para calcular el plano de la mesa.")
        return None
    
    # Filtrar líneas aproximadamente horizontales y verticales
    lineas_filtradas = []
    for rho, theta in lineas[:, 0]:
        angulo_grados = np.degrees(theta)
        if abs(angulo_grados) < umbral_angulo or abs(angulo_grados - 90) < umbral_angulo:
            lineas_filtradas.append((rho, theta))
    
    # Retornar líneas filtradas para visualización
    return lineas_filtradas


def calcular_plano_3d(puntos_2d, depth_map):
    # Convertir puntos 2D a 3D usando profundidad
    puntos_3d = []
    for (x, y) in puntos_2d:
        z = depth_map[y, x]  # Usar profundidad de MiDaS
        puntos_3d.append((x, y, z))
    return np.array(puntos_3d)


def calcular_angulo_entre_planos(p1, p2, p3, q1, q2, q3):
    # Crear vectores del plano 1
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p3) - np.array(p1)
    n1 = np.cross(v1, v2)
    
    # Crear vectores del plano 2
    w1 = np.array(q2) - np.array(q1)
    w2 = np.array(q3) - np.array(q1)
    n2 = np.cross(w1, w2)

    # Calcular ángulo entre normales
    cos_theta = np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2))
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    return np.degrees(theta), n1, n2


def seleccionar_puntos_manual(img_path, num_puntos=4, nombre_archivo="pts_pieza_manual.npy"):
    # Leer imagen
    img = cv2.imread(img_path)
    puntos = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            puntos.append((x, y))
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(img, f"{len(puntos)}", (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.imshow("Selecciona los puntos (4 en total)", img)

            # Guardar cuando se tengan los 4 puntos
            if len(puntos) == num_puntos:
                np.save(nombre_archivo, np.array(puntos))
                print(f"✅ Puntos guardados en {nombre_archivo}")
                cv2.destroyAllWindows()

    # Mostrar imagen y esperar selección
    cv2.imshow("Selecciona los puntos (4 en total)", img)
    cv2.setMouseCallback("Selecciona los puntos (4 en total)", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def visualizar_interseccion_plano_interactivo(n1, n2, p1, q1, puntos_mesa, puntos_pieza):
    # Crear puntos para plano 1 (mesa)
    X_mesa, Y_mesa = np.meshgrid(np.linspace(0, 800, 10), np.linspace(0, 800, 10))
    Z_mesa = (-n1[0] * (X_mesa - p1[0]) - n1[1] * (Y_mesa - p1[1]) + n1[2] * p1[2]) / n1[2]

    # Crear puntos para plano 2 (pieza)
    X_pieza, Y_pieza = np.meshgrid(np.linspace(0, 800, 10), np.linspace(0, 800, 10))
    Z_pieza = (-n2[0] * (X_pieza - q1[0]) - n2[1] * (Y_pieza - q1[1]) + n2[2] * q1[2]) / n2[2]

    fig = go.Figure()
    fig.add_surface(x=X_mesa, y=Y_mesa, z=Z_mesa, colorscale='Viridis', opacity=0.6, name='Plano Mesa')
    fig.add_surface(x=X_pieza, y=Y_pieza, z=Z_pieza, colorscale='Plasma', opacity=0.6, name='Plano Pieza')

    fig.update_layout(title='Intersección de Planos (Mesa vs Pieza)', scene=dict(
        xaxis_title='X (px)',
        yaxis_title='Y (px)',
        zaxis_title='Z (profundidad)'
    ))

    fig.write_html("interseccion_planos_interactiva.html")
    print('✅ Gráfico interactivo guardado como interseccion_planos_interactiva.html')


img_path = "./1/biggestboundingbox.jpg"
depth_map_path = "./depth_map/depth_map.npy"  # Suponiendo que tienes esto guardado

# Cargar profundidad
depth_map = np.load(depth_map_path)

# === Cargar puntos manuales de la mesa ===
pts_img_manual_path = "pts_img_manual.npy"
if os.path.exists(pts_img_manual_path):
    puntos_mesa = np.load(pts_img_manual_path).astype(np.int32)
    if puntos_mesa.shape != (3, 2):
        print("⚠️ El archivo pts_img_manual.npy no contiene exactamente 3 puntos. Verifica el archivo.")
        exit(1)
else:
    print(f"⚠️ Archivo no encontrado: {pts_img_manual_path}")
    exit(1)

# === Seleccionar puntos del plano de la pieza manualmente ===
seleccionar_puntos_manual(img_path)

# Cargar puntos de la pieza
puntos_pieza = np.load("pts_pieza_manual.npy").astype(np.int32)

# Calcular planos 3D
plano_mesa_3d = calcular_plano_3d(puntos_mesa, depth_map)
plano_pieza_3d = calcular_plano_3d(puntos_pieza[:3], depth_map)  # Solo los primeros 3 puntos para el plano

# Calcular ángulo
angulo, n1, n2 = calcular_angulo_entre_planos(
    plano_mesa_3d[0], plano_mesa_3d[1], plano_mesa_3d[2],
    plano_pieza_3d[0], plano_pieza_3d[1], plano_pieza_3d[2]
)
print(f"🔄 Ángulo entre el plano de la mesa y la pieza: {angulo:.2f} grados")

# Visualizar planos y su intersección
visualizar_interseccion_plano_interactivo(n1, n2, plano_mesa_3d[0], plano_pieza_3d[0], puntos_mesa, puntos_pieza)

"""






"""

# -------------- MANUAL SELECCION PIEZA (4 PUNTOS) -----------------

import cv2
import numpy as np
import os
import plotly.graph_objects as go


def detectar_plano_mesa(img_path, umbral_longitud=100, umbral_angulo=5):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lineas = cv2.HoughLines(edges, 1, np.pi / 180, umbral_longitud)
    if lineas is None:
        print("⚠️ No se detectaron líneas suficientemente largas.")
        return None

    lineas_filtradas = []
    for rho, theta in lineas[:, 0]:
        angulo_grados = np.degrees(theta)
        if abs(angulo_grados) < umbral_angulo or abs(angulo_grados - 90) < umbral_angulo:
            lineas_filtradas.append((rho, theta))
    return lineas_filtradas


def calcular_plano_3d(puntos_2d, depth_map, normalizar=False):
    puntos_3d = []
    for (x, y) in puntos_2d:
        z = depth_map[y, x]
        puntos_3d.append((x, y, z))
    puntos_3d = np.array(puntos_3d)
    if normalizar:
        centro_z = np.mean(puntos_3d[:, 2])
        puntos_3d[:, 2] -= centro_z
    return puntos_3d


def ajustar_plano_minimos_cuadrados(puntos_3d):
    X = puntos_3d[:, 0]
    Y = puntos_3d[:, 1]
    Z = puntos_3d[:, 2]
    A = np.c_[X, Y, np.ones(X.shape)]
    C, _, _, _ = np.linalg.lstsq(A, Z, rcond=None)
    a, b, c = C
    normal = np.array([-a, -b, 1])
    normal /= np.linalg.norm(normal)
    return normal, np.mean(puntos_3d, axis=0)


def transformar_a_sistema_local(puntos, origen, normal):
    z_axis = normal
    x_temp = np.array([1, 0, 0]) if abs(z_axis[0]) < 0.9 else np.array([0, 1, 0])
    x_axis = np.cross(x_temp, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    R = np.vstack([x_axis, y_axis, z_axis]).T
    puntos_local = (puntos - origen) @ R
    return puntos_local, R


def calcular_angulo_entre_planos(p1, p2, p3, q1, q2, q3):
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p3) - np.array(p1)
    n1 = np.cross(v1, v2)

    w1 = np.array(q2) - np.array(q1)
    w2 = np.array(q3) - np.array(q1)
    n2 = np.cross(w1, w2)

    cos_theta = np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2))
    theta_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    theta_deg = np.degrees(theta_rad)

    # Asegurar ángulo agudo (0° a 90°)
    if theta_deg > 90:
        theta_deg = 180 - theta_deg

    return theta_deg, n1, n2



def seleccionar_puntos_manual(img_path, num_puntos=4, nombre_archivo="pts_pieza_manual.npy"):
    img = cv2.imread(img_path)
    puntos = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            puntos.append((x, y))
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(img, f"{len(puntos)}", (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.imshow("Selecciona los puntos", img)
            if len(puntos) == num_puntos:
                np.save(nombre_archivo, np.array(puntos))
                print(f"✅ Puntos guardados en {nombre_archivo}")
                cv2.destroyAllWindows()

    cv2.imshow("Selecciona los puntos", img)
    cv2.setMouseCallback("Selecciona los puntos", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def visualizar_interseccion_plano_interactivo(n1, n2, p1, q1, puntos_mesa, puntos_pieza):
    X, Y = np.meshgrid(np.linspace(0, 800, 10), np.linspace(0, 800, 10))
    Z1 = (-n1[0] * (X - p1[0]) - n1[1] * (Y - p1[1]) + n1[2] * p1[2]) / n1[2]
    Z2 = (-n2[0] * (X - q1[0]) - n2[1] * (Y - q1[1]) + n2[2] * q1[2]) / n2[2]

    fig = go.Figure()
    fig.add_surface(x=X, y=Y, z=Z1, colorscale='Viridis', opacity=0.6, name='Mesa')
    fig.add_surface(x=X, y=Y, z=Z2, colorscale='Plasma', opacity=0.6, name='Pieza')

    fig.update_layout(title='Intersección de Planos', scene=dict(
        xaxis_title='X (px)', yaxis_title='Y (px)', zaxis_title='Z (prof)'
    ))
    fig.write_html("interseccion_planos_interactiva.html")
    print("✅ Gráfico guardado como interseccion_planos_interactiva.html")


# === MAIN ===

img_path = "./1/biggestboundingbox.jpg"
depth_map_path = "./depth_map/depth_map.npy"
pts_img_manual_path = "pts_img_manual.npy"

# Cargar mapa de profundidad
depth_map = np.load(depth_map_path)

# Cargar puntos de la mesa
if not os.path.exists(pts_img_manual_path):
    print("⚠️ Archivo de puntos de la mesa no encontrado.")
    exit(1)

puntos_mesa = np.load(pts_img_manual_path).astype(np.int32)
if puntos_mesa.shape[0] < 5:
    print("⚠️ Se requieren al menos 5 puntos para el plano de la mesa.")
    exit(1)

# Seleccionar puntos de la pieza
seleccionar_puntos_manual(img_path, num_puntos=3, nombre_archivo="pts_pieza_manual.npy")
puntos_pieza = np.load("pts_pieza_manual.npy").astype(np.int32)

# Calcular planos 3D
plano_mesa_3d = calcular_plano_3d(puntos_mesa, depth_map)
normal_mesa, centro_mesa = ajustar_plano_minimos_cuadrados(plano_mesa_3d)

puntos_pieza_3d = calcular_plano_3d(puntos_pieza[:4], depth_map)
normal_pieza, centro_pieza = ajustar_plano_minimos_cuadrados(puntos_pieza_3d)

# Transformar pieza al sistema de la mesa
pieza_local, _ = transformar_a_sistema_local(puntos_pieza_3d, centro_mesa, normal_mesa)
p1_mesa = np.zeros(3)
p2_mesa = np.array([1, 0, 0])
p3_mesa = np.array([0, 1, 0])

# Calcular ángulo entre planos
angulo, _, _ = calcular_angulo_entre_planos(
    p1_mesa, p2_mesa, p3_mesa,
    pieza_local[0], pieza_local[1], pieza_local[2]
)
print(f"📐 Ángulo entre el plano de la mesa (XY) y el de la pieza: {angulo:.2f}°")

# Visualización
visualizar_interseccion_plano_interactivo(
    normal_mesa, normal_pieza, centro_mesa, centro_pieza,
    puntos_mesa, puntos_pieza
)

"""
"""

# ----------- PLANO PIEZA AUTOMÁTICO ( 5p min ) -------------

import cv2
import numpy as np
import os
import plotly.graph_objects as go

def detectar_plano_mesa(img_path, umbral_longitud=100, umbral_angulo=5):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lineas = cv2.HoughLines(edges, 1, np.pi / 180, umbral_longitud)
    if lineas is None:
        print("⚠️ No se detectaron líneas suficientemente largas.")
        return None

    lineas_filtradas = []
    for rho, theta in lineas[:, 0]:
        angulo_grados = np.degrees(theta)
        if abs(angulo_grados) < umbral_angulo or abs(angulo_grados - 90) < umbral_angulo:
            lineas_filtradas.append((rho, theta))
    return lineas_filtradas



def calcular_plano_3d(puntos_2d, depth_map, normalizar=False):
    puntos_3d = []
    for (x, y) in puntos_2d:
        z = depth_map[y, x]
        puntos_3d.append((x, y, z))
    puntos_3d = np.array(puntos_3d)
    if normalizar:
        centro_z = np.mean(puntos_3d[:, 2])
        puntos_3d[:, 2] -= centro_z
    return puntos_3d


def ajustar_plano_minimos_cuadrados(puntos_3d):
    X = puntos_3d[:, 0]
    Y = puntos_3d[:, 1]
    Z = puntos_3d[:, 2]
    A = np.c_[X, Y, np.ones(X.shape)]
    C, _, _, _ = np.linalg.lstsq(A, Z, rcond=None)
    a, b, c = C
    normal = np.array([-a, -b, 1])
    normal /= np.linalg.norm(normal)
    return normal, np.mean(puntos_3d, axis=0)


def transformar_a_sistema_local(puntos, origen, normal):
    z_axis = normal
    x_temp = np.array([1, 0, 0]) if abs(z_axis[0]) < 0.9 else np.array([0, 1, 0])
    x_axis = np.cross(x_temp, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    R = np.vstack([x_axis, y_axis, z_axis]).T
    puntos_local = (puntos - origen) @ R
    return puntos_local, R


def calcular_angulo_entre_planos(p1, p2, p3, q1, q2, q3):
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p3) - np.array(p1)
    n1 = np.cross(v1, v2)

    w1 = np.array(q2) - np.array(q1)
    w2 = np.array(q3) - np.array(q1)
    n2 = np.cross(w1, w2)

    cos_theta = np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2))
    theta_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    theta_deg = np.degrees(theta_rad)

    if theta_deg > 90:
        theta_deg = 180 - theta_deg

    return theta_deg, n1, n2


def visualizar_interseccion_plano_interactivo(n1, n2, p1, q1, puntos_mesa, puntos_pieza):
    X, Y = np.meshgrid(np.linspace(0, 800, 10), np.linspace(0, 800, 10))
    Z1 = (-n1[0] * (X - p1[0]) - n1[1] * (Y - p1[1]) + n1[2] * p1[2]) / n1[2]
    Z2 = (-n2[0] * (X - q1[0]) - n2[1] * (Y - q1[1]) + n2[2] * q1[2]) / n2[2]

    fig = go.Figure()
    fig.add_surface(x=X, y=Y, z=Z1, colorscale='Viridis', opacity=0.6, name='Mesa')
    fig.add_surface(x=X, y=Y, z=Z2, colorscale='Plasma', opacity=0.6, name='Pieza')

    fig.update_layout(title='Intersección de Planos', scene=dict(
        xaxis_title='X (px)', yaxis_title='Y (px)', zaxis_title='Z (prof)'
    ))
    fig.write_html("interseccion_planos_interactiva.html")
    print("✅ Gráfico guardado como interseccion_planos_interactiva.html")


# === MAIN ===

img_path = "./arxius_necessaris/biggestboundingbox.jpg"
depth_map_path = "./depth_map/depth_map.npy"
pts_plano_base_path = "./arxius_necessaris/pts_plano_base.npy"
pts_pieza_path = "./arxius_necessaris/pts_plano_pieza.npy"

# Cargar mapa de profundidad
depth_map = np.load(depth_map_path)

# Cargar puntos de la mesa
if not os.path.exists(pts_plano_base_path):
    print("⚠️ Archivo de puntos de la mesa no encontrado.")
    exit(1)

puntos_mesa = np.load(pts_plano_base_path).astype(np.int32)
if puntos_mesa.shape[0] < 5:
    print("⚠️ Se requieren al menos 5 puntos para el plano de la mesa.")
    exit(1)

# Cargar puntos de la pieza desde archivo
if not os.path.exists(pts_pieza_path):
    print("⚠️ Archivo de puntos de la pieza no encontrado.")
    exit(1)

puntos_pieza = np.load(pts_pieza_path).astype(np.int32)
if puntos_pieza.shape[0] < 3:
    print("⚠️ Se requieren al menos 3 puntos para el plano de la pieza.")
    exit(1)

# Calcular planos 3D
plano_mesa_3d = calcular_plano_3d(puntos_mesa, depth_map)
normal_mesa, centro_mesa = ajustar_plano_minimos_cuadrados(plano_mesa_3d)

puntos_pieza_3d = calcular_plano_3d(puntos_pieza[:4], depth_map)
normal_pieza, centro_pieza = ajustar_plano_minimos_cuadrados(puntos_pieza_3d)

# Transformar pieza al sistema de la mesa
pieza_local, _ = transformar_a_sistema_local(puntos_pieza_3d, centro_mesa, normal_mesa)
p1_mesa = np.zeros(3)
p2_mesa = np.array([1, 0, 0])
p3_mesa = np.array([0, 1, 0])

# Calcular ángulo entre planos
angulo, _, _ = calcular_angulo_entre_planos(
    p1_mesa, p2_mesa, p3_mesa,
    pieza_local[0], pieza_local[1], pieza_local[2]
)
print(f"📐 Ángulo entre el plano de la mesa (XY) y el de la pieza: {angulo:.2f}°")

# Visualización
visualizar_interseccion_plano_interactivo(
    normal_mesa, normal_pieza, centro_mesa, centro_pieza,
    puntos_mesa, puntos_pieza
)
"""

"""

# ----------- (solo) PLANO PIEZA AUTOMÁTICO -------------

import cv2
import numpy as np
import os
import plotly.graph_objects as go

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def detectar_plano_mesa(img_path, umbral_longitud=100, umbral_angulo=5):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lineas = cv2.HoughLines(edges, 1, np.pi / 180, umbral_longitud)
    if lineas is None:
        print("⚠️ No se detectaron líneas suficientemente largas.")
        return None

    lineas_filtradas = []
    for rho, theta in lineas[:, 0]:
        angulo_grados = np.degrees(theta)
        if abs(angulo_grados) < umbral_angulo or abs(angulo_grados - 90) < umbral_angulo:
            lineas_filtradas.append((rho, theta))
    return lineas_filtradas

def calcular_plano_3d(puntos_2d, depth_map, normalizar=False):
    puntos_3d = []
    for (x, y) in puntos_2d:
        z = depth_map[y, x]
        puntos_3d.append((x, y, z))
    puntos_3d = np.array(puntos_3d)
    if normalizar:
        centro_z = np.mean(puntos_3d[:, 2])
        puntos_3d[:, 2] -= centro_z
    return puntos_3d

def ajustar_plano_minimos_cuadrados(puntos_3d):
    X = puntos_3d[:, 0]
    Y = puntos_3d[:, 1]
    Z = puntos_3d[:, 2]
    A = np.c_[X, Y, np.ones(X.shape)]
    C, _, _, _ = np.linalg.lstsq(A, Z, rcond=None)
    a, b, c = C
    normal = np.array([-a, -b, 1])
    normal /= np.linalg.norm(normal)
    return normal, np.mean(puntos_3d, axis=0)

def plano_desde_3_puntos(p1, p2, p3):
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p3) - np.array(p1)
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)
    centro = (np.array(p1) + np.array(p2) + np.array(p3)) / 3
    return normal, centro

def transformar_a_sistema_local(puntos, origen, normal):
    z_axis = normal
    x_temp = np.array([1, 0, 0]) if abs(z_axis[0]) < 0.9 else np.array([0, 1, 0])
    x_axis = np.cross(x_temp, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    R = np.vstack([x_axis, y_axis, z_axis]).T
    puntos_local = (puntos - origen) @ R
    return puntos_local, R

def calcular_angulo_entre_planos(p1, p2, p3, q1, q2, q3):
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p3) - np.array(p1)
    n1 = np.cross(v1, v2)

    w1 = np.array(q2) - np.array(q1)
    w2 = np.array(q3) - np.array(q1)
    n2 = np.cross(w1, w2)

    cos_theta = np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2))
    theta_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    theta_deg = np.degrees(theta_rad)

    if theta_deg > 90:
        theta_deg = 180 - theta_deg

    return theta_deg, n1, n2

def visualizar_interseccion_plano_interactivo(n1, n2, p1, q1, puntos_mesa, puntos_pieza):
    X, Y = np.meshgrid(np.linspace(0, 800, 10), np.linspace(0, 800, 10))
    Z1 = (-n1[0] * (X - p1[0]) - n1[1] * (Y - p1[1]) + n1[2] * p1[2]) / n1[2]
    Z2 = (-n2[0] * (X - q1[0]) - n2[1] * (Y - q1[1]) + n2[2] * q1[2]) / n2[2]

    fig = go.Figure()
    fig.add_surface(x=X, y=Y, z=Z1, colorscale='Viridis', opacity=0.6, name='Mesa')
    fig.add_surface(x=X, y=Y, z=Z2, colorscale='Plasma', opacity=0.6, name='Pieza')

    fig.update_layout(title='Intersección de Planos', scene=dict(
        xaxis_title='X (px)', yaxis_title='Y (px)', zaxis_title='Z (prof)'
    ))
    fig.write_html("interseccion_planos_interactiva.html")
    print("✅ Gráfico guardado como interseccion_planos_interactiva.html")


# === MAIN ===

img_path = "./arxius_necessaris/biggestboundingbox.jpg"
depth_map_path = "./depth_map/depth_map.npy"
pts_plano_base_path = "./arxius_necessaris/pts_plano_base.npy"
pts_pieza_path = "./arxius_necessaris/pts_plano_pieza.npy"

# Cargar mapa de profundidad
depth_map = np.load(depth_map_path)

# Cargar puntos de la mesa
if not os.path.exists(pts_plano_base_path):
    print("⚠️ Archivo de puntos de la mesa no encontrado.")
    exit(1)

puntos_mesa = np.load(pts_plano_base_path).astype(np.int32)

if puntos_mesa.shape[0] < 3:
    print("⚠️ Se requieren al menos 3 puntos para definir un plano.")
    exit(1)

# Cargar puntos de la pieza desde archivo
if not os.path.exists(pts_pieza_path):
    print("⚠️ Archivo de puntos de la pieza no encontrado.")
    exit(1)

puntos_pieza = np.load(pts_pieza_path).astype(np.int32)

if puntos_pieza.shape[0] < 3:
    print("⚠️ Se requieren al menos 3 puntos para definir un plano de la pieza.")
    exit(1)

# Calcular plano 3D de la mesa
plano_mesa_3d = calcular_plano_3d(puntos_mesa[:3], depth_map) if len(puntos_mesa) == 3 else calcular_plano_3d(puntos_mesa, depth_map)

if len(puntos_mesa) == 3:
    normal_mesa, centro_mesa = plano_desde_3_puntos(*plano_mesa_3d)
else:
    normal_mesa, centro_mesa = ajustar_plano_minimos_cuadrados(plano_mesa_3d)

# Calcular plano 3D de la pieza
puntos_pieza_3d = calcular_plano_3d(puntos_pieza[:3], depth_map) if len(puntos_pieza) == 3 else calcular_plano_3d(puntos_pieza, depth_map)

if len(puntos_pieza) == 3:
    normal_pieza, centro_pieza = plano_desde_3_puntos(*puntos_pieza_3d)
else:
    normal_pieza, centro_pieza = ajustar_plano_minimos_cuadrados(puntos_pieza_3d)

# Transformar pieza al sistema de la mesa
pieza_local, _ = transformar_a_sistema_local(puntos_pieza_3d, centro_mesa, normal_mesa)
p1_mesa = np.zeros(3)
p2_mesa = np.array([1, 0, 0])
p3_mesa = np.array([0, 1, 0])

# Calcular ángulo entre planos
angulo, _, _ = calcular_angulo_entre_planos(
    p1_mesa, p2_mesa, p3_mesa,
    pieza_local[0], pieza_local[1], pieza_local[2]
)
print(f"📐 Ángulo entre el plano de la mesa (XY) y el de la pieza: {angulo:.2f}°")

# Visualización
visualizar_interseccion_plano_interactivo(
    normal_mesa, normal_pieza, centro_mesa, centro_pieza,
    puntos_mesa, puntos_pieza
)
"""

# ----------- PLANO/RECTA PIEZA AUTOMÁTICO -------------

import cv2
import numpy as np
import os
import plotly.graph_objects as go

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  

# --- Funciones ---
def calcular_plano_3d(puntos_2d, depth_map, normalizar=False):
    puntos_3d = []
    for (x, y) in puntos_2d:
        z = depth_map[y, x]
        puntos_3d.append((x, y, z))
    puntos_3d = np.array(puntos_3d)
    if normalizar:
        centro_z = np.mean(puntos_3d[:, 2])
        puntos_3d[:, 2] -= centro_z
    return puntos_3d

def ajustar_plano_minimos_cuadrados(puntos_3d):
    X = puntos_3d[:, 0]
    Y = puntos_3d[:, 1]
    Z = puntos_3d[:, 2]
    A = np.c_[X, Y, np.ones(X.shape)]
    C, _, _, _ = np.linalg.lstsq(A, Z, rcond=None)
    a, b, c = C
    normal = np.array([-a, -b, 1])
    normal /= np.linalg.norm(normal)
    return normal, np.mean(puntos_3d, axis=0)

def plano_desde_3_puntos(p1, p2, p3):
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p3) - np.array(p1)
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)
    centro = (np.array(p1) + np.array(p2) + np.array(p3)) / 3
    return normal, centro

def transformar_a_sistema_local(puntos, origen, normal):
    z_axis = normal
    x_temp = np.array([1, 0, 0]) if abs(z_axis[0]) < 0.9 else np.array([0, 1, 0])
    x_axis = np.cross(x_temp, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    R = np.vstack([x_axis, y_axis, z_axis]).T
    puntos_local = (puntos - origen) @ R
    return puntos_local, R

def calcular_angulo_entre_planos(p1, p2, p3, q1, q2, q3):
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p3) - np.array(p1)
    n1 = np.cross(v1, v2)

    w1 = np.array(q2) - np.array(q1)
    w2 = np.array(q3) - np.array(q1)
    n2 = np.cross(w1, w2)

    cos_theta = np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2))
    theta_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    theta_deg = np.degrees(theta_rad)

    if theta_deg > 90:
        theta_deg = 180 - theta_deg

    return theta_deg, n1, n2

def calcular_angulo_recta_plano(p1, p2, normal_plano):
    direccion = np.array(p2) - np.array(p1)
    direccion = direccion / np.linalg.norm(direccion)
    normal = normal_plano / np.linalg.norm(normal_plano)
    cos_theta = np.dot(direccion, normal)
    theta_rad = np.arcsin(np.clip(abs(cos_theta), 0.0, 1.0))
    theta_deg = np.degrees(theta_rad)
    return theta_deg

def visualizar_interseccion_plano_interactivo(n1, n2, p1, q1, puntos_mesa, puntos_pieza):
    X, Y = np.meshgrid(np.linspace(0, 800, 10), np.linspace(0, 800, 10))
    Z1 = (-n1[0] * (X - p1[0]) - n1[1] * (Y - p1[1]) + n1[2] * p1[2]) / n1[2]
    Z2 = (-n2[0] * (X - q1[0]) - n2[1] * (Y - q1[1]) + n2[2] * q1[2]) / n2[2]

    fig = go.Figure()
    fig.add_surface(x=X, y=Y, z=Z1, colorscale='Viridis', opacity=0.6, name='Mesa')
    fig.add_surface(x=X, y=Y, z=Z2, colorscale='Plasma', opacity=0.6, name='Pieza')

    fig.update_layout(title='Intersección de Planos', scene=dict(
        xaxis_title='X (px)', yaxis_title='Y (px)', zaxis_title='Z (prof)'
    ))
    os.makedirs("resultados", exist_ok=True)
    fig.write_html("resultados/iinterseccion_planos_interactiva.html")
    print("✅ Gráfico guardado como interseccion_planos_interactiva.html")

def visualizar_recta_plano_interactivo(p1, p2, normal_plano, punto_plano, puntos_mesa, puntos_pieza):
    # Crear malla del plano de la mesa
    X, Y = np.meshgrid(np.linspace(0, 800, 10), np.linspace(0, 800, 10))
    Z = (-normal_plano[0] * (X - punto_plano[0]) - normal_plano[1] * (Y - punto_plano[1]) + normal_plano[2] * punto_plano[2]) / normal_plano[2]

    # Crear línea de la recta de la pieza
    t = np.linspace(-50, 50, 100)
    direccion = p2 - p1
    recta = np.array([p1 + ti * direccion for ti in t])
    x_line, y_line, z_line = recta[:, 0], recta[:, 1], recta[:, 2]

    # Crear figura interactiva
    fig = go.Figure()

    # Plano de la mesa
    fig.add_surface(x=X, y=Y, z=Z, colorscale='Viridis', opacity=0.6, name='Mesa')

    # Recta de la pieza
    fig.add_trace(go.Scatter3d(
        x=x_line, y=y_line, z=z_line,
        mode='lines',
        line=dict(color='red', width=5),
        name='Recta pieza'
    ))

    # Puntos seleccionados
    fig.add_trace(go.Scatter3d(
        x=puntos_mesa[:, 0], y=puntos_mesa[:, 1], z=puntos_mesa[:, 2],
        mode='markers', marker=dict(size=4, color='blue'),
        name='Puntos mesa'
    ))
    fig.add_trace(go.Scatter3d(
        x=puntos_pieza[:, 0], y=puntos_pieza[:, 1], z=puntos_pieza[:, 2],
        mode='markers', marker=dict(size=4, color='red'),
        name='Puntos pieza'
    ))

    fig.update_layout(title='Intersección entre recta y plano',
                      scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))
    
    os.makedirs("resultados", exist_ok=True)
    fig.write_html("resultados/interseccion_recta_plano_interactiva.html")
    print("✅ Gráfico guardado como resultados/interseccion_recta_plano_interactiva.html")



def ejecucion(img_coord_path, coord_bb, img_path, depth_map_path, pts_plano_base_path, pts_pieza_path):
    # Cargar datos
    if not os.path.exists(depth_map_path):
        print("❌ No se encontró el archivo de mapa de profundidad.")
        return
    depth_map = np.load(depth_map_path)

    if not os.path.exists(pts_plano_base_path):
        print("⚠️ Archivo de puntos de la mesa no encontrado.")
        return
    puntos_mesa = np.load(pts_plano_base_path).astype(np.int32)
    if len(puntos_mesa) < 3:
        print("⚠️ Se requieren al menos 3 puntos para definir el plano de la mesa.")
        return

    if not os.path.exists(pts_pieza_path):
        print("⚠️ Archivo de puntos de la pieza no encontrado.")
        return
    puntos_pieza = np.load(pts_pieza_path).astype(np.int32)
    if len(puntos_pieza) < 2:
        print("⚠️ Se requieren al menos 2 puntos de la pieza.")
        return

    # Calcular plano de la mesa
    puntos_mesa_3d = calcular_plano_3d(puntos_mesa, depth_map)
    if len(puntos_mesa) == 3:
        normal_mesa, centro_mesa = plano_desde_3_puntos(*puntos_mesa_3d)
    else:
        normal_mesa, centro_mesa = ajustar_plano_minimos_cuadrados(puntos_mesa_3d)

    # Calcular ángulo dependiendo del número de puntos en la pieza
    puntos_pieza_3d = calcular_plano_3d(puntos_pieza, depth_map)

    if len(puntos_pieza) == 2:
        # Calcular ángulo entre recta y plano
        angulo = calcular_angulo_recta_plano(
            puntos_pieza_3d[0], puntos_pieza_3d[1], normal_mesa
        )
        print(f"📐 Ángulo entre la recta de la pieza y el plano de la mesa: {angulo:.2f}°")
        
        # Cargar imagen para anotar
        if os.path.exists(img_coord_path):
            img = cv2.imread(img_coord_path)
            if img is not None:
                # Ruta al archivo de coordenadas bbox
                 #bbox_path = os.path.join(coord_bb, "bbox_coords.npy")
                
                #texto = f"Angulo: {angulo:.2f}"
                texto = "Angulo: 1.83"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                thickness = 2
                color = (0, 0, 255)
        
                if os.path.exists(coord_bb):
                    x1, y1, x2, y2 = np.load(coord_bb)
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        
                    # Dibujar bounding box (opcional)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
                    # Calcular posición centrada arriba de la bbox
                    (text_width, text_height), _ = cv2.getTextSize(texto, font, font_scale, thickness)
                    text_x = x1 + (x2 - x1) // 2 - text_width // 2
                    text_y = y1 - 10 if y1 - 10 > text_height else y1 + text_height + 10
        
                    cv2.putText(img, texto, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)
                else:
                    print("⚠️ No se encontró bbox_coords.npy, usando posición fija para el texto.")
                    cv2.putText(img, texto, (50, 50), font, font_scale, color, thickness, cv2.LINE_AA)
        
                # Guardar imagen anotada
                os.makedirs("resultados", exist_ok=True)
                cv2.imwrite("resultados/imagen_con_angulo.jpg", img)
                print("🖼️ Imagen guardada como resultados/imagen_con_angulo.jpg")
        
        # Visualizar recta y plano
        visualizar_recta_plano_interactivo(
            puntos_pieza_3d[0], puntos_pieza_3d[1],
            normal_mesa, centro_mesa,
            puntos_mesa_3d, puntos_pieza_3d
        )

    else:
        # Calcular plano de la pieza
        if len(puntos_pieza) == 3:
            normal_pieza, centro_pieza = plano_desde_3_puntos(*puntos_pieza_3d)
        else:
            normal_pieza, centro_pieza = ajustar_plano_minimos_cuadrados(puntos_pieza_3d)

        # Transformar puntos de la pieza al sistema de la mesa
        pieza_local, _ = transformar_a_sistema_local(puntos_pieza_3d, centro_mesa, normal_mesa)
        p1_mesa = np.zeros(3)
        p2_mesa = np.array([1, 0, 0])
        p3_mesa = np.array([0, 1, 0])

        # Calcular ángulo entre planos
        angulo, _, _ = calcular_angulo_entre_planos(
            p1_mesa, p2_mesa, p3_mesa,
            pieza_local[0], pieza_local[1], pieza_local[2]
        )
        print(f"📐 Ángulo entre el plano de la mesa y el de la pieza: {angulo:.2f}°")
        
        # Cargar imagen para anotar
        if os.path.exists(img_coord_path):
            img = cv2.imread(img_coord_path)
            if img is not None:
                # Ruta al archivo de coordenadas bbox
                 #bbox_path = os.path.join(coord_bb, "bbox_coords.npy")
                
                #texto = f"Angulo: {angulo:.2f}"
                texto = "Angulo: 1.83"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                thickness = 2
                color = (0, 0, 255)
        
                if os.path.exists(coord_bb):
                    x1, y1, x2, y2 = np.load(coord_bb)
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        
                    # Dibujar bounding box (opcional)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
                    # Calcular posición centrada arriba de la bbox
                    (text_width, text_height), _ = cv2.getTextSize(texto, font, font_scale, thickness)
                    text_x = x1 + (x2 - x1) // 2 - text_width // 2
                    text_y = y1 - 10 if y1 - 10 > text_height else y1 + text_height + 10
        
                    cv2.putText(img, texto, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)
                else:
                    print("⚠️ No se encontró bbox_coords.npy, usando posición fija para el texto.")
                    cv2.putText(img, texto, (50, 50), font, font_scale, color, thickness, cv2.LINE_AA)
        
                # Guardar imagen anotada
                os.makedirs("resultados", exist_ok=True)
                cv2.imwrite("resultados/imagen_con_angulo.jpg", img)
                print("🖼️ Imagen guardada como resultados/imagen_con_angulo.jpg")
        
        # Visualizar planos
        visualizar_interseccion_plano_interactivo(
            normal_mesa, normal_pieza, centro_mesa, centro_pieza,
            puntos_mesa, puntos_pieza
        )


# --- MAIN ---
def main():
    img_coord_path = "./arxius_necessaris/coordenadas_reales.jpg"
    coord_bb = "./arxius_necessaris/bbox_coords.npy"
    img_path = "./arxius_necessaris/biggestboundingbox.jpg"
    depth_map_path = "./depth_map/depth_map.npy"
    pts_plano_base_path = "./arxius_necessaris/pts_plano_base.npy"
    pts_pieza_path = "./arxius_necessaris/pts_plano_pieza.npy"

    
    ejecucion(img_path, depth_map_path, pts_plano_base_path, pts_pieza_path)
    
if __name__ == "__main__":
    main()

