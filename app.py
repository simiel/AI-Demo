import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import random

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))


@flask_app.route("/")
def index():
    return render_template("index.html", f1='', f2='', f3='', f4='',customer_name='Sir/Madam')


@flask_app.route("/predict", methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    return render_template("index.html", prediction_text="The Lifetime monetary value of this customer is {} Cedis".format(prediction))


@flask_app.route("/generate", methods=["POST"])
def generate():
    f1 = random.randint(5, 20)
    f2 = random.randint(18, 70)
    f3 = random.randint(1, 2)
    f4 = random.randint(1, 4)
    list1 = ['Chris', 'Vanessa','Abigail','Nii','Nana','Ohemaa','Simiel','LSK']
    list2 = ['Mensah','Kumi','Pompow','Asare','Inkoom','Boakye','Koranteng']
    name1= random.choice(list1)
    name2= random.choice(list2)
    return render_template("index.html", prediction_text='Waiting', f1=f1, f2=f2, f3=f3, f4=f4,customer_name=name1+' '+name2)


if __name__ == "__main__":
    flask_app.run(debug=True)
