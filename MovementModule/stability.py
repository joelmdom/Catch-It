"""
Algoritmos para calcular si la torre es estable. Usa vision por computador.
"""
import cv2
import numpy as np
from math import atan2, degrees

class StabilityModule():
    def __init__(self):
        pass

    def calculate_stability(self, img_path):
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detectar bordes
        edges = cv2.Canny(gray, 50, 150)

        # Cortar solo la franja central donde está la torre
        h, w = edges.shape
        x_min = int(w * 0.4)
        x_max = int(w * 0.6)
        roi = edges[:, x_min:x_max]

        # Calcular el centro de masa horizontal por fila
        points = []
        for y in range(h):
            row = roi[y]
            xs = np.where(row > 0)[0]
            if len(xs) > 0:
                x_mean = np.mean(xs) + x_min  # Reajustar a coordenadas globales
                points.append((int(x_mean), y))
                # print(f"centro de masa: {x_mean}, {y}")

        # Si hay suficientes puntos, ajustar una línea
        if len(points) > 10:
            pts = np.array(points)
            x_coords = pts[:, 0]
            y_coords = pts[:, 1]

            # Ajustar línea por mínimos cuadrados
            A = np.vstack([y_coords, np.ones_like(y_coords)]).T
            m, c = np.linalg.lstsq(A, x_coords, rcond=None)[0]

            # Calcular ángulo (con respecto a la vertical)
            angle_rad = atan2(m, 1)
            angle_deg = degrees(angle_rad)

            # dibujar linea para debug output
            output = img.copy()
            y1, y2 = 0, h
            x1 = int(m * y1 + c)
            x2 = int(m * y2 + c)
            cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(output, f"Inclinacion: {angle_deg:.2f}°", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        else:
            output = img.copy()
            cv2.putText(output, "No se encontraron suficientes puntos", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Mostrar por pantalla
        cv2.imshow("Debug View", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

import glob
if __name__ == '__main__':
    stability = StabilityModule()

    for img in glob.glob("./img/*.png"):
        stability.calculate_stability(img)
