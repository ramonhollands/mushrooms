from fastai import *
from fastai.vision import *
from fastai.callbacks import *
from pathlib import Path
import torch
import scipy.ndimage
import base64
import uuid


from flask import Flask, render_template, request, jsonify
from io import BytesIO
import os

path = Path(".")
keywords = 'champignon,oesterzwam,tropische beurszwam,shiitake,morielje,cantharel,gewoon eekhoorntjesbrood,Truffels,zwavelzwam,geschubde inktzwam,gewone fluweelpootje,gordijnzwam,doodstrompet,Anijschampignon,Reuzenchampignon,Appelrussula,Biefstukzwam,Gele stekelzwam,Gewone fopzwam,Grote parasolzwam,Grote sponszwam,Grote stinkzwam,Judasoor,Kastanjeboleet,Knolparasolzwam,Koeienboleet,Paarse schijnridderzwam,Paarssteelschijnridderzwam,Parelamaniet,Pruikzwam,Regenboogrussula,Reuzenbovist,Roodbruine slanke amaniet,Voorjaarspronkridder,Weidechampignon,Zwavelzwam,Smakelijke russula,Zwartwitte veldridderzwam,Parelhoenchampignon,Karbolchampignon,Narcisamaniet,Vliegenzwam,Panteramaniet,Groene knolamaniet,Porfieramaniet,Voorjaarsamaniet,Kleverige knolamaniet,Netstelige heksenboleet,Satansboleet,Witte trechterzwammen,Witte bundelridderzwam,Grote bostrechterzwam,Grote kale inktzwam,Berkenzwam,Gordijnzwammen,Vermiljoengordijnzwam,Pagemantel,Satijnzwam,Bundelmosklokje,Prachtvlamhoed,Voorjaarskluifzwam,Radijsvaalhoed,Witte kluifzwam,Zwarte kluifzwam,Gewone zwavelkop,Vezelkoppen,Sterspoorvezelkop,Giftige vezelkop,Witte satijnvezelkop,Zandpad vezelkop,Geelbruine spleetvezelkop,Parasolzwammen,Spitsschubbige parasolzwam,Kastanjeparasolzwam,Gewoon elfenschermpje,Zwartbruine vlekplaat,Grauwe vlekplaat,Gazonvlekplaat,Gewone krulzoom,Grauwgroene hertenzwam,Kaalkopjes,Puntig kaalkopje,Fraaie koraalzwam,Duivelsbroodrussula,Braakrussula,Blauwvoetstekelzwam,Kroonbekerzwam,Kleine aardappelbovist,Gele aardappelbovist,Wortelende aardappelbovist,Oranje ridderzwam,Gele ridderzwam,Narcisridderzwam,Beukenridderzwam'
classes = keywords.split(',')

data = ImageDataBunch.single_from_classes(path, classes, tfms=get_transforms(), size=224).normalize(imagenet_stats)
learn = create_cnn(data, models.resnet34, metrics=accuracy)
learn.model.eval()
learn.load('all_incl_latin_resnet34_defaults_424_')

mr_df = pd.read_csv(path/'mushrooms.csv')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('upload.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route("/upload", methods=["POST"])
def upload():
    bytes = request.files['file'].read()
    return predict_image_from_bytes(bytes)

def get_class_info(class_index, classes, certainty, mr_df):
    class_name = classes[class_index]
    row = mr_df.loc[mr_df['name'] == class_name]
    latin_name = row.values[0][1]
    wiki_link = row.values[0][2]
    eatable = row.values[0][3]
    # example_image = os.listdir(path/'static'/'images'/class_name)[0]

    eatable_class = 'not_eatable'
    if(eatable):
        eatable_class = 'eatable'

    return {
        'wiki_link': wiki_link,
        'example_image': ['aaa.jpg', 'bbb.jpg'],
        'eatable_class': eatable_class,
        'class_name' : class_name,
        'latin_name': latin_name,
        'certainty': certainty
    }

# https://github.com/henripal/maps_webapp/blob/master/model_backend/cities.py
def predict_image_from_bytes(bytes):
    img = open_image(BytesIO(bytes))
    (class_name, class_index, certainties) = learn.predict(img)
    certainties = certainties.numpy()
    # print(certainties)
    n = 3
    top_three = np.argsort(certainties)[::-1][:n]
    predictions = [get_class_info(i, classes, certainties[i], mr_df) for i in top_three]

    if(certainties[top_three[:1]] < 0.4):
        predict_html = render_template('no-mushroom.html',
                           predictions=predictions)
        title = "Uhmm, that's not a mushroom"
        subtitle = ''
        return jsonify({
            'predict_html': predict_html,
            'title': predictions[0]['latin_name'],
            'subtitle': 'Let me show your some examples'})

    random_user_file_name = str(uuid.uuid4().hex)
    heatmap_from_img(img, random_user_file_name)

    predict_html = render_template('prediction.html',
                           predictions=predictions,
                           heatmap=random_user_file_name)

    example_html = render_template('examples.html', predictions=predictions)

    lookalikes_html = render_template('lookalikes.html', predictions=predictions)

    return jsonify({
                    'predict_html': predict_html,
                    'title': 'Looks like the ' + predictions[0]['latin_name'],
                    'subtitle': 'Please check the example images below to be sure',
                    'example_html': example_html,
                    'lookalikes_html' : lookalikes_html
    })

def heatmap_from_img(img, random_user_file_name):
    img.save('static/user_images/'+random_user_file_name+'.png')
    img = img.resize(224)

    # pred_class, pred_idx, outputs = learn.predict(img)
    img = img.px.reshape(1, 3, 224, 224)

    upsampled = run_gradcam(img)
    gbp_map = run_gbp(img)

    upsampled_to_b64bytes(upsampled, gbp_map, random_user_file_name)

def upsampled_to_b64bytes(upsampled, img, random_user_file_name):
    """
    this combines upsampled heatmap and img
    and returns b64 encoded bytes for the image
    """
    figfile = BytesIO()

    fig = plt.figure(frameon=False)
    fig.set_size_inches(2,2)

    # all this to remove borders
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    combined = np.einsum('ijk, ij->ijk',img, upsampled)
    ax.imshow(combined)
    plt.savefig('static/user_images/'+random_user_file_name+'_heatmap.png')

    # figfile.seek(0)
    # figdata_png = base64.b64encode(figfile.getvalue())

    # return figdata_png

def run_gradcam(img):
    """
    returns the heatmap for the given image
    """
    # last bottleneck module
    learn.model.eval()
    target_layer = learn.model[0][7][2]

    fmap_hook, gradient_hook = create_hooks(target_layer)
    run_backprop_once(img)

    gradient  = next(iter(gradient_hook.stored))
    linearization = gradient.cpu().numpy().sum((2, 3)).reshape(-1)
    fmaps = fmap_hook.stored.cpu().numpy()
    fmaps = fmaps.reshape(512, 7, 7)

    hm = np.maximum(0, np.einsum('i, ijk',linearization, fmaps))
    upsampled = scipy.ndimage.zoom(hm, 32)

    return normalize_image(upsampled)

def create_hooks(target_layer):
    fmap_hook = callbacks.hook_output(target_layer)
    gradient_hook = callbacks.Hook(target_layer, gradient_torch_hook, is_forward=False)

    return fmap_hook, gradient_hook

def run_backprop_once(img):
    learn.model.zero_grad()
    # forward
    out = learn.model(img)

    # gradient wrt the predicted class only
    onehot = torch.zeros(learn.data.c)
    onehot[torch.argmax(out)] = 1.0

    # backwrd
    out.backward(gradient=onehot.reshape(1, -1))

def gradient_torch_hook(self, grad_input, grad_output):
    return grad_input

def normalize_image(image):
    return (image-np.min(image))/(np.max(image)-np.min(image))

def run_gbp(img):
    learn.model.eval()
    create_gp_hooks()
    img.requires_grad_()
    run_backprop_once(img)

    return image_from_tensor(img.grad)

def create_gp_hooks():
    relu_modules = [module[1] for module in learn.model.named_modules() if str(module[1]) == "ReLU(inplace)"]
    hooks = Hooks(relu_modules, clamp_gradients_hook, is_forward=False)

def clamp_gradients_hook(module, grad_in, grad_out):
    for grad in grad_in:
        torch.clamp_(grad, min=0.0)

def image_from_tensor(imagetensor):
    numpied = torch.squeeze(imagetensor)
    numpied = np.moveaxis(numpied.detach().cpu().numpy(), 0 , -1)
    return normalize_image(numpied)

if __name__ == '__main__':
    if "serve" in sys.argv:
        port = int(os.environ.get("PORT", 8008))
        app.run(debug=True, host='0.0.0.0', port=port)