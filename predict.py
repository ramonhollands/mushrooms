from fastai import *
from fastai.vision import *
import torch
from pathlib import Path

from flask import Flask, render_template, request
from io import BytesIO
import os

# path = Path('/Users/ramon/AI/mushrooms')
path = Path(".")
keywords = 'champignon,oesterzwam,gewoon eekhoorntjesbrood'
classes = keywords.split(',')
#
# import os
# # cwd = os.getcwd()
# print(os.listdir(path))

data = ImageDataBunch.single_from_classes(path, classes, tfms=get_transforms(), size=224).normalize(imagenet_stats)
learn = create_cnn(data, models.resnet34, metrics=accuracy)
learn.model.eval()
learn.load('3_resnet34_defaults')

# img = open_image(path/'images'/'gewoon eekhoorntjesbrood'/'eekhoorntjesbrood.jpg')
# (class_name, nr_of_classes, losses) = learn.predict(img)

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('upload.html')

@app.route("/upload", methods=["POST"])
def upload():
    bytes = request.files['file'].read()
    return predict_image_from_bytes(bytes)

# @app.route("/")
# def form(request):
#     env = Environment(
#         loader=PackageLoader('static', 'templates'),
#         autoescape=select_autoescape(['html', 'xml'])
#     )
#     template = env.get_template('upload.html')
#     return HTMLResponse(template.render())

def predict_image_from_bytes(bytes):
    img = open_image(BytesIO(bytes))
    (class_name, nr_of_classes, losses) = learn.predict(img)
    example_image = os.listdir(path/'static'/'images'/class_name)[0]
    # strings = ["%.2f" % loss for loss in losses.numpy()]
    return 'Looks like the "'+class_name+'":<br /><br /><img src="/static/images/'+class_name+'/'+example_image+'" />'

# def get_view_predictions(class_name, losses):
#     return HTMLResponse(class_name + '<br/>' + '<br />'.join(losses))
#
# if __name__ == "__main__":
#     if "serve" in sys.argv:
#         port = int(os.environ.get("PORT", 8008))
#         uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8008))
    app.run(debug=True, host='0.0.0.0', port=port)