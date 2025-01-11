from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from PIL import Image
from backend.model import handle_dicom, ndarray_to_image, get_model

import io
import os
import numpy as np
from contextlib import asynccontextmanager


model = None

predict_dir = "predictions"
os.makedirs(predict_dir, exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = get_model(512)  # Создаем модель один раз при запуске
    print("Модель получена")
    yield
    
app = FastAPI(lifespan=lifespan)

# Указываем папку со статическими файлами
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
def read_root():
    return FileResponse("frontend/index.html")

# Маршрут для обработки изображений
@app.post("/predict_liver/")
async def process_image(file: UploadFile = File(...)):
    image_file = await file.read()
    file_extension = file.filename.split(".")[-1].lower()
    print(file_extension)
    
    if file_extension not in ["dcm", "tiff"]:
        return JSONResponse(
        content={"error": "Unsupported file format. Please upload DICOM (.dcm) or TIFF (.tiff) files."},
        status_code=400
    )
    
    is_dicom = file_extension == "dcm"
    tiff_image = handle_dicom(image_file, is_dicom)
    
    prediction = model.predict(tiff_image)
    
    pred_io = ndarray_to_image(prediction)
    
    return StreamingResponse(pred_io, media_type="image/png")

@app.post("/save_edited_mask")
async def save_edited_mask(file: UploadFile = File(...)):
    from backend.model import file_name
    file_path = file_name + "_predict.png"
    file_location = os.path.join(predict_dir, file_path)
    with open(file_location, "wb") as f:
        f.write(await file.read())

        return {"message": "Mask saved successfully."}
    
@app.get("/get_image")
async def get_dicom_image():
    from backend.model import file_name
    img = tiff_to_png(file_name)
    return StreamingResponse(img, media_type="image/png")

def tiff_to_png(file_name):
    tiff_img = Image.open(f"{predict_dir}/{file_name}.tiff")
    image_array = np.array(tiff_img)
    
    # Нормализация пикселей в диапазон 0–255
    min_val = np.min(image_array)
    max_val = np.max(image_array)
    normalized_array = np.clip((image_array - min_val) / (max_val - min_val) * 255, 0, 255).astype(np.uint8)
    
    # Преобразуем обратно в изображение
    normalized_image = Image.fromarray(normalized_array)
    
    # Сохраняем в байтовый поток как PNG
    img_io = io.BytesIO()
    normalized_image.save(img_io, format="PNG")
    img_io.seek(0)
    return img_io