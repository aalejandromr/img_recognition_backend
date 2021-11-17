from flask import Flask, request
import os
from werkzeug.utils import secure_filename
from flask_cors import CORS
import tensorflow as tf
import numpy as np

app = Flask(__name__)
CORS(app)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = f"{ROOT_DIR}/images_to_predict"
CLASS_NAMES = ['birthday cake', 'mailbox', 'waterslide']
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/predict', methods=['POST'])
def predict():
  if request.method == 'POST':
    file = request.files['file']
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    img_height = 180
    img_width = 180
    img = tf.keras.utils.load_img(
        os.path.join(app.config['UPLOAD_FOLDER'], filename), target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    model = tf.keras.models.load_model('my_model.h5')
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    return(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(CLASS_NAMES[np.argmax(score)], 100 * np.max(score))
    )

@app.route('/')
def index():
  return "<h1>Welcome to CodingX</h1>"