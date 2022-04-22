import pickle
import numpy as np
import tensorflow as tf
import cv2
from flask import Flask, render_template, request, redirect, flash
from werkzeug.utils import secure_filename
import os

UPLOAD_FOLDER = 'uploads'
app = Flask(__name__, static_url_path='/static',
            static_folder='./static',
            template_folder='templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model_image = tf.keras.models.load_model('Covid_19_detection.h5')
with open('./model_risk.pkl', 'rb') as model_p:
    model_risk = pickle.load(model_p)

with open('./model_symptoms.pkl', 'rb') as model_p:
    model_symptoms = pickle.load(model_p)


@app.route('/')
def root():
    return render_template('index.html')


@app.route('/index.html')
def index():
    return render_template('index.html')


@app.route('/contact.html')
def contact():
    return render_template('contact.html')


@app.route('/about.html')
def about():
    return render_template('about.html')


@app.route('/faqs.html')
def faqs():
    return render_template('faqs.html')


@app.route('/prevention.html')
def prevention():
    return render_template('prevention.html')


@app.route('/upload.html')
def upload():
    return render_template('upload.html')


@app.route('/upload_ct.html')
def upload_ct():
    return render_template('upload_ct.html')


@app.route('/form.html')
def form():
    return render_template('form.html')


@app.route('/predictor', methods=['POST'])
def predictor():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224))
            images_final = np.array(image) / 255
            pred_1 = model_image.predict(np.array([images_final]))

            if (np.round(pred_1[0][0])) > 0.5:
                msg = "High Risk."
                return render_template("results_final.html", message=msg)
            elif 0.5 > (np.round(pred_1[0][0])) > 0.3:
                msg = "Moderate Risk."
                return render_template("results_final.html", message=msg)
            else:
                msg = "No Issues Detected."
                return render_template("results_final.html", message=msg)


@app.route('/re', methods=['POST'])
def re():
    if request.method == 'POST':
        return render_template('form.html')


@app.route('/getinfo', methods=['POST'])
def getinfo():
    fields = ["name", "age", "gender", "pneumonia", "stiffness", "effusion", "tiredness", "thirst", "loss_of_appetite",
              "weakness",
              "discomfort", "cough", "body_pains", "rigor", "cold", "sore_throat", "breathlessness",
              "fever", "headache", "unable_to_taste"]
    record = []
    feat = []
    if request.method == 'POST':
        for i in range(len(fields)):
            if request.form.get(fields[i]) == "":
                flash('fields missing')
                return redirect(request.url)

            record.append(request.form.get(fields[i]))
        feat.append(record[0])

        if record[2].upper() == 'M' or record[2].upper() == 'MALE':
            feat.append(1)
        else:
            feat.append(0)
        feat.append(record[1])
        i = 3
        while i < len(fields):
            if record[i].upper() == 'Y' or record[i].upper() == 'YES':
                feat.append(1)
            else:
                feat.append(0)
            i += 1
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            image = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224))
            images_final = np.array(image) / 255
            pred_1 = model_image.predict(np.array([images_final]))
            pred = pred_1[0][0]
        else:
            flash('image missing')
            return redirect(request.url)

        train_1 = np.array([feat[1], float(feat[2]), pred])
        train_2 = [int(feat[1]), float(feat[2])]
        i = 3
        while i < len(feat):
            train_2.append(int(feat[i]))
            i += 1
        train_2 = np.array(train_2)
        print(train_1)
        print(train_2)
        test = [1, 65.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        test = np.array(test)
        pred_risks = model_risk.predict(np.array([train_1]))
        pred_symptoms = model_symptoms.predict(np.array([test]))

        overall_pred = (float(pred_risks[0]) + float(pred_symptoms[0])) / 2.0
        if overall_pred > 0.5:
            msg = "Status: Risky. Acute Symptoms detected. Advised to take immediate Medical Help."
        else:
            msg = "Status: Covid Not Detected. Nothing to worry about."
        return render_template("results_final_form.html", message=msg, paitentName=record[0])


if __name__ == '__main__':
    app.run()
