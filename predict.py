from fastai import *
from fastai.vision import *
import torch
from pathlib import Path

from flask import Flask, render_template, request
from io import BytesIO
import os

path = Path(".")
keywords = 'champignon,oesterzwam,tropische beurszwam,shiitake,morielje,cantharel,gewoon eekhoorntjesbrood,Truffels,zwavelzwam,geschubde inktzwam,gewone fluweelpootje,gordijnzwam,doodstrompet,Anijschampignon,Reuzenchampignon,Appelrussula,Biefstukzwam,Gele stekelzwam,Gewone fopzwam,Grote parasolzwam,Grote sponszwam,Grote stinkzwam,Judasoor,Kastanjeboleet,Knolparasolzwam,Koeienboleet,Paarse schijnridderzwam,Paarssteelschijnridderzwam,Parelamaniet,Pruikzwam,Regenboogrussula,Reuzenbovist,Roodbruine slanke amaniet,Voorjaarspronkridder,Weidechampignon,Zwavelzwam,Smakelijke russula,Zwartwitte veldridderzwam,Parelhoenchampignon,Karbolchampignon,Narcisamaniet,Vliegenzwam,Panteramaniet,Groene knolamaniet,Porfieramaniet,Voorjaarsamaniet,Kleverige knolamaniet,Netstelige heksenboleet,Satansboleet,Witte trechterzwammen,Witte bundelridderzwam,Grote bostrechterzwam,Grote kale inktzwam,Berkenzwam,Gordijnzwammen,Vermiljoengordijnzwam,Pagemantel,Satijnzwam,Bundelmosklokje,Prachtvlamhoed,Voorjaarskluifzwam,Radijsvaalhoed,Witte kluifzwam,Zwarte kluifzwam,Gewone zwavelkop,Vezelkoppen,Sterspoorvezelkop,Giftige vezelkop,Witte satijnvezelkop,Zandpad vezelkop,Geelbruine spleetvezelkop,Parasolzwammen,Spitsschubbige parasolzwam,Kastanjeparasolzwam,Gewoon elfenschermpje,Zwartbruine vlekplaat,Grauwe vlekplaat,Gazonvlekplaat,Gewone krulzoom,Grauwgroene hertenzwam,Kaalkopjes,Puntig kaalkopje,Fraaie koraalzwam,Duivelsbroodrussula,Braakrussula,Blauwvoetstekelzwam,Kroonbekerzwam,Kleine aardappelbovist,Gele aardappelbovist,Wortelende aardappelbovist,Oranje ridderzwam,Gele ridderzwam,Narcisridderzwam,Beukenridderzwam'
classes = keywords.split(',')

data = ImageDataBunch.single_from_classes(path, classes, tfms=get_transforms(), size=224).normalize(imagenet_stats)
learn = create_cnn(data, models.resnet34, metrics=accuracy)
learn.model.eval()
learn.load('resnet34_defaults_all_classes')

mr_df = pd.read_csv(path/'mushrooms.csv')

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('upload.html')

@app.route("/upload", methods=["POST"])
def upload():
    bytes = request.files['file'].read()
    return predict_image_from_bytes(bytes)

def predict_image_from_bytes(bytes):
    img = open_image(BytesIO(bytes))
    (class_name, nr_of_classes, losses) = learn.predict(img)

    # TODO
    # get 2nd and 3th class
    # print them as well

    row = mr_df.loc[mr_df['name'] == class_name]
    latin_name = row.values[0][1]
    wiki_link = row.values[0][2]
    eatable = row.values[0][3]
    example_image = os.listdir(path/'static'/'images'/class_name)[0]

    eatable_class = 'not_eatable'
    if(eatable):
        eatable_class = 'eatable'

    # strings = ["%.2f" % loss for loss in losses.numpy()]
    return render_template('prediction.html',
                           wiki_link=wiki_link, latin_name=latin_name,
                           example_image=example_image, class_name=class_name, eatable_class=eatable_class,
                           wiki_link2=wiki_link, latin_name2=latin_name,
                           example_image2=example_image, class_name2=class_name, eatable_class2=eatable_class)

if __name__ == '__main__':
    if "serve" in sys.argv:
        port = int(os.environ.get("PORT", 8008))
        app.run(debug=True, host='0.0.0.0', port=port)