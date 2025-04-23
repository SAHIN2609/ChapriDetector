from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import os

app = FastAPI()

model = tf.keras.models.load_model(os.path.join("..", "model", "chapri_model.h5"))

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = image.resize((1024, 1024))
    img_array = np.expand_dims(np.array(image) / 255.0, axis=0)

    prediction = model.predict(img_array)[0][0]

    result = "CHAPRI 😎" if prediction > 0.5 else "NOT CHAPRI 👌"
    return JSONResponse(content={"result": result})
