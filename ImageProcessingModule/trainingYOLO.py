# -*- coding: utf-8 -*-
"""
Autors: Lucía Torrescusa Rubio y Joel Montes de Oca    
"""

from ultralytics import YOLO
import os


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Cargar modelo base YOLOv8n
model = YOLO('yolov8n.pt')
#plot_labels = lambda *a, **kw: None
# Entrenar el modelo
model.train(
    plots=False,
    data='data.yaml',  
    epochs=50,
    imgsz=640,
    batch=8
)