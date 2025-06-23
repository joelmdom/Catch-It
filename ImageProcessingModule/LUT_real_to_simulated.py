"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.exposure import match_histograms

# ----------- HISTOGRAM MATCHING -----------

# Cargar imágenes
real = cv2.imread("real.jpg")
sim = cv2.imread("simulada.png")

# Verificar que se cargaron correctamente
if real is None or sim is None:
    raise FileNotFoundError("No se pudieron cargar las imágenes.")

# Redimensionar 'real' al tamaño de 'simulada' (parece que querías eso)
#real = cv2.resize(real, (sim.shape[1], sim.shape[0]))

# Aplicar igualación de histogramas
matched = match_histograms(real, sim, channel_axis=-1)

# Convertir imágenes de BGR a RGB para visualizar con matplotlib
real_rgb = cv2.cvtColor(real, cv2.COLOR_BGR2RGB)
sim_rgb = cv2.cvtColor(sim, cv2.COLOR_BGR2RGB)
matched_rgb = cv2.cvtColor(matched.astype(np.uint8), cv2.COLOR_BGR2RGB)

plt.imshow(real_rgb)
plt.axis('off')
plt.show()

plt.imshow(sim_rgb)
plt.axis('off')
plt.show()

plt.imshow(matched_rgb)
plt.axis('off')
plt.show()

# ------------------------------
"""

"""
# ----------------- MEDIA IMAGENES --> LUT --------------------------

import os
import cv2
import numpy as np

print("Directorio actual:", os.getcwd())

def obtener_imagenes_en_orden(carpeta):
    extensiones = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    archivos = [f for f in os.listdir(carpeta) if os.path.splitext(f)[1].lower() in extensiones]
    return sorted(archivos)

# Definir las rutas de las carpetas
carpeta1 = "./reales/"
carpeta2 = "./simuladas/"

imagenes_carpeta1 = obtener_imagenes_en_orden(carpeta1)
imagenes_carpeta2 = obtener_imagenes_en_orden(carpeta2)

# Emparejar imágenes con rutas completas
pares = []
for img1, img2 in zip(imagenes_carpeta1, imagenes_carpeta2):
    ruta1 = os.path.join(carpeta1, img1)
    ruta2 = os.path.join(carpeta2, img2)
    if os.path.exists(ruta1) and os.path.exists(ruta2):
        pares.append((ruta1, ruta2))
    else:
        print(f" Archivo faltante: {ruta1} o {ruta2}")

def calcular_lut_por_canal(real_channel, sim_channel):
    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        indices = np.where(real_channel == i)
        if len(indices[0]) > 0:
            valores_sim = sim_channel[indices]
            lut[i] = np.clip(np.mean(valores_sim), 0, 255).astype(np.uint8)
        else:
            lut[i] = i
    return lut

def entrenar_luts(pares_real_simulada):
    lut_b_total = []
    lut_g_total = []
    lut_r_total = []

    for real_path, sim_path in pares_real_simulada:
        real = cv2.imread(real_path)
        sim = cv2.imread(sim_path)

        if real is None:
            print(f"No se pudo cargar la imagen real: {real_path}")
            continue
        if sim is None:
            print(f"No se pudo cargar la imagen simulada: {sim_path}")
            continue
        
        #real = cv2.resize(real, (sim.shape[1], sim.shape[0]))
        
        # ---------- CLAHE - RGB ------------------

        lab = cv2.cvtColor(real, cv2.COLOR_BGR2LAB)

        lab_planes = list(cv2.split(lab))

        clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))

        lab_planes[0] = clahe.apply(lab_planes[0])

        lab = cv2.merge(lab_planes)

        real = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # ---------------------------------------

        b_real, g_real, r_real = cv2.split(real)
        b_sim, g_sim, r_sim = cv2.split(sim)

        lut_b_total.append(calcular_lut_por_canal(b_real, b_sim))
        lut_g_total.append(calcular_lut_por_canal(g_real, g_sim))
        lut_r_total.append(calcular_lut_por_canal(r_real, r_sim))

    if not lut_b_total or not lut_g_total or not lut_r_total:
        raise ValueError("No se pudieron entrenar LUTs. Verifica que las imágenes se hayan cargado correctamente.")

    # Promedio final
    lut_b = np.mean(lut_b_total, axis=0).astype(np.uint8)
    lut_g = np.mean(lut_g_total, axis=0).astype(np.uint8)
    lut_r = np.mean(lut_r_total, axis=0).astype(np.uint8)

    return lut_b, lut_g, lut_r

def aplicar_luts(imagen_real, lut_b, lut_g, lut_r):
    if imagen_real is None:
        raise ValueError("Imagen real no cargada correctamente.")
    b, g, r = cv2.split(imagen_real)
    b_corr = cv2.LUT(b, lut_b)
    g_corr = cv2.LUT(g, lut_g)
    r_corr = cv2.LUT(r, lut_r)
    return cv2.merge([b_corr, g_corr, r_corr])

# Entrenar LUTs
lut_b, lut_g, lut_r = entrenar_luts(pares)

# Aplicar a una nueva imagen real
nueva_real_path = "nueva_real.jpg"
nueva_real = cv2.imread(nueva_real_path)

if nueva_real is None:
    raise FileNotFoundError(f" No se pudo cargar la nueva imagen real: {nueva_real_path}")

nueva_corregida = aplicar_luts(nueva_real, lut_b, lut_g, lut_r)

# Mostrar o guardar
cv2.imshow("Corregida", nueva_corregida)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ------------------------------


"""

"""

# --------------- TRANSFERENCIA REINHARD (1 img) ------------------------

import cv2
import numpy as np

def transferencia_reinhard(im_real, im_simulada):
    # Convertimos las imágenes a Lab
    lab_real = cv2.cvtColor(im_real, cv2.COLOR_BGR2Lab).astype(np.float32)
    lab_sim = cv2.cvtColor(im_simulada, cv2.COLOR_BGR2Lab).astype(np.float32)

    # Separamos los canales
    l_real, a_real, b_real = cv2.split(lab_real)
    l_sim, a_sim, b_sim = cv2.split(lab_sim)

    # Función para ajustar un canal
    def ajustar_canal(c_real, c_sim):
        media_real, std_real = c_real.mean(), c_real.std()
        media_sim, std_sim = c_sim.mean(), c_sim.std()
        c_ajustada = (c_real - media_real) * (std_sim / std_real) + media_sim
        return np.clip(c_ajustada, 0, 255)

    # Ajustamos los canales
    l_final = ajustar_canal(l_real, l_sim)
    a_final = ajustar_canal(a_real, a_sim)
    b_final = ajustar_canal(b_real, b_sim)

    # Fusionamos y convertimos de nuevo a BGR
    lab_final = cv2.merge([l_final, a_final, b_final]).astype(np.uint8)
    bgr_final = cv2.cvtColor(lab_final, cv2.COLOR_Lab2BGR)

    return bgr_final


# Cargar imágenes
real = cv2.imread("./nueva_real.jpg")
#simulada = cv2.imread("./simulada.png")
simulada = cv2.imread("./simulada_solapada.png")


# DEFINIR AREA A RECORTAR IMAGEN

# Asegurar que tengan el mismo tamaño
simulada = cv2.resize(simulada, (real.shape[1], real.shape[0]))

cv2.imshow("Transferencia Reinhard", simulada)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Aplicar transferencia
real_adaptada = transferencia_reinhard(real, simulada)

# Guardar o mostrar
cv2.imwrite("real_adaptada_2.jpg", real_adaptada)
cv2.imshow("Transferencia Reinhard", real_adaptada)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ------------------------------

"""

"""

# ---------- TRANSFERENCIA REINHARD (multi) ---------------
# - media y desviación estándar promedio

import cv2
import numpy as np
import glob

def calcular_estadisticas_lab(imagenes_lab):
    # Acumular estadísticas
    medias = []
    stds = []

    for lab in imagenes_lab:
        l, a, b = cv2.split(lab)
        medias.append([l.mean(), a.mean(), b.mean()])
        stds.append([l.std(), a.std(), b.std()])

    # Convertir a numpy arrays para facilitar el cálculo
    medias = np.array(medias)
    stds = np.array(stds)

    # Media de medias y media de desviaciones estándar
    media_global = medias.mean(axis=0)
    std_global = stds.mean(axis=0)

    return media_global, std_global

def transferencia_reinhard_multi(im_real, imagenes_simuladas):
    lab_real = cv2.cvtColor(im_real, cv2.COLOR_BGR2Lab).astype(np.float32)

    # Convertir todas las imágenes simuladas a Lab y redimensionarlas
    imagenes_lab = []
    for im in imagenes_simuladas:
        im = cv2.resize(im, (im_real.shape[1], im_real.shape[0]))
        imagenes_lab.append(cv2.cvtColor(im, cv2.COLOR_BGR2Lab).astype(np.float32))

    # Obtener estadísticas globales
    media_sim, std_sim = calcular_estadisticas_lab(imagenes_lab)

    # Estadísticas de la imagen real
    l_real, a_real, b_real = cv2.split(lab_real)
    media_real = np.array([l_real.mean(), a_real.mean(), b_real.mean()])
    std_real = np.array([l_real.std(), a_real.std(), b_real.std()])

    # Aplicar la transformación
    l = (l_real - media_real[0]) * (std_sim[0] / std_real[0]) + media_sim[0]
    a = (a_real - media_real[1]) * (std_sim[1] / std_real[1]) + media_sim[1]
    b = (b_real - media_real[2]) * (std_sim[2] / std_real[2]) + media_sim[2]

    # Reconstruir y convertir a BGR
    lab_final = cv2.merge([np.clip(l, 0, 255), np.clip(a, 0, 255), np.clip(b, 0, 255)]).astype(np.uint8)
    return cv2.cvtColor(lab_final, cv2.COLOR_Lab2BGR)

# Cargar imagen real
real = cv2.imread("./nueva_real.jpg")

# Cargar múltiples imágenes simuladas (ajusta el patrón según tus archivos)
rutas_simuladas = glob.glob("./train/*.png")  # o .jpg
imagenes_simuladas = [cv2.imread(ruta) for ruta in rutas_simuladas]

# Aplicar transferencia
real_adaptada = transferencia_reinhard_multi(real, imagenes_simuladas)

# Guardar o mostrar
cv2.imwrite("real_adaptada_multi.jpg", real_adaptada)
cv2.imshow("Transferencia Reinhard Promediada", real_adaptada)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""

"""
# ------------ TRANSFORMACION REINHARD --> PRECALCULO (tipo LUT) - 1 img real ----------------


import cv2
import numpy as np
import glob
import pickle

def calcular_estadisticas_lab(imagenes_lab):
    medias = []
    stds = []
    for lab in imagenes_lab:
        l, a, b = cv2.split(lab)
        medias.append([l.mean(), a.mean(), b.mean()])
        stds.append([l.std(), a.std(), b.std()])
    return np.mean(medias, axis=0), np.mean(stds, axis=0)

def precalcular_transformacion(simuladas_paths, real_path, guardar_en="transform_reinhard.pkl"):
    real = cv2.imread(real_path)
    lab_real = cv2.cvtColor(real, cv2.COLOR_BGR2Lab).astype(np.float32)

    imagenes_lab = []
    for path in simuladas_paths:
        img = cv2.imread(path)
        img = cv2.resize(img, (real.shape[1], real.shape[0]))
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab).astype(np.float32)
        imagenes_lab.append(lab)

    media_sim, std_sim = calcular_estadisticas_lab(imagenes_lab)

    l_real, a_real, b_real = cv2.split(lab_real)
    media_real = np.array([l_real.mean(), a_real.mean(), b_real.mean()])
    std_real = np.array([l_real.std(), a_real.std(), b_real.std()])

    # Calcular escala y desplazamiento para aplicar transformación directamente
    scale = std_sim / std_real
    shift = media_sim - media_real * scale

    transform = {
        "scale": scale.tolist(),
        "shift": shift.tolist()
    }

    with open(guardar_en, "wb") as f:
        pickle.dump(transform, f)

    print(f"Transformación tipo LUT funcional guardada en {guardar_en}")


# CODIGO TRANSFORMACION IMAGEN CON "LUT" PRECALCULADA

def aplicar_transformacion(im_real, transform_path="transform_reinhard.pkl"):
    with open(transform_path, "rb") as f:
        transform = pickle.load(f)

    lab = cv2.cvtColor(im_real, cv2.COLOR_BGR2Lab).astype(np.float32)
    l, a, b = cv2.split(lab)

    canales_in = [l, a, b]
    canales_out = []

    for i in range(3):
        canal_transformado = canales_in[i] * transform["scale"][i] + transform["shift"][i]
        canales_out.append(np.clip(canal_transformado, 0, 255))

    lab_out = cv2.merge(canales_out).astype(np.uint8)
    return cv2.cvtColor(lab_out, cv2.COLOR_Lab2BGR)


# 1: precalcular la "LUT"
simuladas_paths = glob.glob("./simuladas/*.png")
precalcular_transformacion(simuladas_paths, "./nueva_real.jpg")

# 2: aplicar a nuevas imágenes
nueva_real = cv2.imread("./nueva_real.jpg")
adaptada = aplicar_transformacion(nueva_real)

cv2.imshow("Resultado", adaptada)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ---------------------------------------------------------------------------

"""

# ------------ TRANSFORMACION REINHARD --> PRECALCULO (tipo LUT) - VARIAS img real ----------------

"""
- No importa que las imágenes tengan resoluciones diferentes. La transformación se basa en estadísticas 
globales (media y desviación estándar) de los canales Lab, no en la posición de los píxeles.

- Solo se ajustan los valores de color, canal por canal (L, a, b), de forma global
- Una imagen real conserva su estructura pero parece tener la misma atmósfera o iluminación que la imagen simulada.
"""


import cv2
import numpy as np
import glob
import json
import matplotlib.pyplot as plt


def calcular_estadisticas_lab(imagenes_lab):
    medias = []
    stds = []
    for lab in imagenes_lab:
        l, a, b = cv2.split(lab)
        medias.append([l.mean(), a.mean(), b.mean()])
        stds.append([l.std(), a.std(), b.std()])
    return np.mean(medias, axis=0), np.mean(stds, axis=0)

def precalcular_transformacion_global(simuladas_paths, reales_paths, guardar_en="transform_reinhard.json"):
    imagenes_sim = []
    imagenes_real = []

    for path in simuladas_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"Advertencia: No se pudo cargar la imagen simulada {path}")
            continue
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab).astype(np.float32)
        imagenes_sim.append(lab)

    for path in reales_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"Advertencia: No se pudo cargar la imagen real {path}")
            continue
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab).astype(np.float32)
        imagenes_real.append(lab)

    if not imagenes_sim or not imagenes_real:
        raise ValueError("No se cargaron suficientes imágenes para calcular la transformación.")

    media_sim, std_sim = calcular_estadisticas_lab(imagenes_sim)
    media_real, std_real = calcular_estadisticas_lab(imagenes_real)

    scale = std_sim / std_real
    shift = media_sim - media_real * scale

    transform = {
        "scale": scale.tolist(),
        "shift": shift.tolist()
    }

    with open(guardar_en, "w") as f:
        json.dump(transform, f)

    print(f"Transformación global guardada en {guardar_en}")

def aplicar_transformacion(im_real, transform_path="transform_reinhard.json"):
    with open(transform_path, "r") as f:
        transform = json.load(f)

    lab = cv2.cvtColor(im_real, cv2.COLOR_BGR2Lab).astype(np.float32)
    l, a, b = cv2.split(lab)

    canales_in = [l, a, b]
    canales_out = []

    for i in range(3):
        canal = canales_in[i] * transform["scale"][i] + transform["shift"][i]
        canales_out.append(np.clip(canal, 0, 255))

    lab_out = cv2.merge(canales_out).astype(np.uint8)
    return cv2.cvtColor(lab_out, cv2.COLOR_Lab2BGR)


def main():
    # Uso
    simuladas_paths = glob.glob("./simuladas/*.png")
    reales_paths = glob.glob("./reales/*.jpg")
    
    precalcular_transformacion_global(simuladas_paths, reales_paths)
    
    nueva_real = cv2.imread("./validacion/val1.jpg")
    if nueva_real is None:
        raise FileNotFoundError("No se pudo cargar './nuevas/real1.jpg'")
    
    adaptada = aplicar_transformacion(nueva_real)
    
    """
    cv2.imshow("Transferencia Reinhard Promedio", adaptada)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
    
    
    plt.imshow(nueva_real[:, :, ::-1])
    plt.axis('off')
    plt.show()
    
    plt.imshow(adaptada[:, :, ::-1])
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()