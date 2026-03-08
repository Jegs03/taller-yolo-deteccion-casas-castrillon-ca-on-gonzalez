# Detector de Casas con YOLOv8 (OBB)

Este proyecto implementa un sistema de detección de objetos orientado a casas utilizando la arquitectura YOLOv8 con soporte para **Oriented Bounding Boxes (OBB)**. El sistema integra un pipeline completo, desde el entrenamiento del modelo hasta un servicio API para inferencias en tiempo real.

## Características
* **Entrenamiento personalizado:** Modelo optimizado con 100 épocas sobre un dataset específico.
* **Inferencia flexible:** Soporte para procesamiento de imágenes mediante rutas locales o carga de archivos en formato bytes.
* **API REST:** Servicio construido con FastAPI para integrar la detección fácilmente en otras aplicaciones.

## Estructura del Proyecto
```text
.
├── data/               # Directorio del dataset (contiene el archivo data.yaml)
├── models/             # Almacena los pesos del modelo entrenado (best.pt)
├── src/                # Lógica de inferencia y utilidades
├── inferencia.py       # API FastAPI para exponer la detección
├── train_yolo.py       # Script de entrenamiento con configuración OBB
└── utils.py            # Funciones auxiliares para el manejo de modelos YOLO
```
## Entrenamiento

El modelo utiliza yolov8m.pt como base. Para entrenar el modelo, se ejecuta train_yolo.py, el cual realiza el entrenamiento y copia automáticamente los mejores pesos (best.pt) al directorio models/ una vez finalizado el proceso.
API de Inferencia

El servicio expone un endpoint para procesar imágenes:

    Endpoint: POST /prediccion/

    Input: Archivo de imagen (UploadFile)

    Output (JSON):

        imagen: Nombre del archivo original.

        detecciones: Lista con clase, confianza y coordenadas rotadas (xyxyxyxy).

        total: Cantidad de casas detectadas.

Ejecución

Para iniciar el servidor de inferencia, utiliza el siguiente comando:
Bash

uvicorn inferencia:app --reload

Requisitos

El proyecto requiere las siguientes dependencias:

    ultralytics

    fastapi

    pillow

    python-multipart
