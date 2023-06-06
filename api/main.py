from fastapi import  FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import cv2
import tensorflow as tf

app = FastAPI()

# endpoint = "http://localhost:8501/v1/models/paddy_model:predict"
MODEL = tf.keras.models.load_model("../saved_models/1")

MODEL = tf.keras.models.load_model("../saved_models/1")
CLASS_NAMES = ['bacterial_leaf_blight',
 'bacterial_leaf_streak',
 'bacterial_panicle_blight',
 'blast',
 'brown_spot',
 'dead_heart',
 'downy_mildew',
 'hispa',
 'normal',
 'tungro']

@app.get("/ping")
async def ping():
    return "Hello I am Alive"


def read_file_as_image(data) ->np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    resized_image = cv2.resize(image, (256, 256))

    img_batch = np.expand_dims(resized_image, 0)

    predictions = MODEL.predict(img_batch)
    predicted_class =   CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)