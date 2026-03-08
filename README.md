# Detector de Casas con YOLOv8 (OBB)

Este proyecto implementa un sistema de detección de objetos orientado a casas utilizando la arquitectura YOLOv8 con soporte para **Oriented Bounding Boxes (OBB)**. El sistema integra un pipeline completo, desde el entrenamiento del modelo hasta un servicio API para inferencias en tiempo real.

## Características
* **Entrenamiento personalizado:** Modelo optimizado con 100 épocas sobre un dataset específico.
* **Inferencia flexible:** Soporte para procesamiento de imágenes mediante rutas locales o carga de archivos en formato bytes.
* **API REST:** Servicio construido con FastAPI para integrar la detección fácilmente en otras aplicaciones.

## Estructura del Proyecto
```text
├── src/
│   ├── train_yolo.py     # Script de entrenamiento (ejecutar CLI o python)
│   ├── inferencia.py     # Script para inferir sobre imágenes nuevas
│   └── utils.py          # Utilidades (convertir formatos, visualizar)
├── models/# Pesos guardados (.pt)
│   └── best.pt
├── requirements.txt      # Dependencias del proyecto
├── data.yaml             # Descriptor del dataset para YOLO
└── README.md
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
