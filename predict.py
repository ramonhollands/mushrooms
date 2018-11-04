from starlette.applications import Starlette
from starlette.responses import JSONResponse, HTMLResponse, RedirectResponse
from fastai import *
from fastai.vision import *
import torch
from pathlib import Path

import uvicorn
import aiohttp
import asyncio

from io import BytesIO

path = Path('/Users/ramon/AI/mushrooms')
keywords = 'champignon,oesterzwam,gewoon eekhoorntjesbrood'
classes = keywords.split(',')

data = ImageDataBunch.single_from_classes(path, classes, tfms=get_transforms(), size=224).normalize(imagenet_stats)
learn = create_cnn(data, models.resnet34, metrics=accuracy)
learn.model.eval();
learn.load('3_resnet34_defaults')

# img = open_image(path/'images'/'eekhoorntjesbrood.jpg')
# results = learn.predict(img)
# print(results)

app = Starlette()

@app.route("/upload", methods=["POST"])
async def upload(request):
    data = await request.form()
    bytes = await (data["file"].read())
    return predict_image_from_bytes(bytes)

@app.route("/")
def form(request):
    return HTMLResponse(
        """
        <form action="/upload" method="post" enctype="multipart/form-data">
            Select image to upload:
            <input type="file" name="file">
            <input type="submit" value="Upload Image">
        </form>
        Or submit a URL:
        <form action="/classify-url" method="get">
            <input type="url" name="url">
            <input type="submit" value="Fetch and analyze image">
        </form>
    """)

def predict_image_from_bytes(bytes):
    img = open_image(BytesIO(bytes))
    (class_name, nr_of_classes, losses) = learn.predict(img)
    return JSONResponse({
        "prediction": class_name,
    })

if __name__ == "__main__":
    if "serve" in sys.argv:
        uvicorn.run(app, host="0.0.0.0", port=8008)
