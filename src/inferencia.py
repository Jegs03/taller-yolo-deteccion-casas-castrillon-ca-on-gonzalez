from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from src.inferencia import inferir_imagen_bytes

app = FastAPI()

@app.post("/prediccion/")
async def prediccion(archivo: UploadFile = File(...)):
    """
    Endpoint que recibe una imagen y realiza la inferencia.
    
    Args:
        archivo: Archivo de imagen subido
    
    Returns:
        Resultados de la inferencia en formato JSON
    """
    # Leer la imagen como bytes
    imagen_bytes = await archivo.read()
    
    # Realizar inferencia
    results = inferir_imagen_bytes(imagen_bytes)
    
    # Procesar resultados
    detecciones = []
    for result in results:
        for box in result.obb:
            detecciones.append({
                "clase": int(box.cls[0]),
                "confianza": float(box.conf[0]),
                "coordenadas": box.xyxyxyxy[0].tolist()
            })
    
    return JSONResponse(content={
        "imagen": archivo.filename,
        "detecciones": detecciones,
        "total": len(detecciones)
    })
