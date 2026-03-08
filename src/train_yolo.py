from ultralytics import YOLO
import shutil
from pathlib import Path

def main():

    model = YOLO("yolov8n-obb.pt")

    results = model.train(
        data="dataset/data.yaml",
        epochs=100,
        imgsz=640,
        batch=8,
        device=0,
        project="runs/obb",
        name="casa_detector"
    )

    best_weights_path = Path(results.save_dir) / "weights" / "best.pt"

    if best_weights_path.exists():
        shutil.copy(best_weights_path, "models/best.pt")
        print("Pesos del modelo guardados en: models/best.pt")
    else:
        print("No se encontraron los pesos del modelo")

    print("Entrenamiento completado!")


if __name__ == "__main__":
    main()
