from flask import Flask, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = f'{ROOT_DIR}/images_to_predict'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/predict', methods=['POST'])
def predict():
  if request.method == 'POST':
    file = request.files['file']
    # if user does not select file, browser also
    # submit a empty part without filename
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))