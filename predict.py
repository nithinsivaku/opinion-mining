#importing libraries
import os
import numpy as np
import flask
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from flask import Flask, render_template, request, jsonify, send_from_directory

# creating instance of the class
app=Flask(__name__)

with open("Models/train.pkl", "rb") as f:
    train_a, train_b = pickle.load(f)
cv=CountVectorizer()
cv.fit_transform(train_a)

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static', 'styles', 'images'),
                               'favicon.ico')

# tell flask what url shoud trigger the function index()
@app.route('/',methods= ["GET", "POST"])
def index():
    if request.method == "POST":
        story =  request.get_json()
        content = story['content']
        print(content)
        if len(content) != 0:
            pred = ValuePredictor(content)
            return jsonify(pred =  pred)
        else:
            return jsonify(pred = 'Input needed'), 500
    return flask.render_template('/index.html')


# prediction function
def ValuePredictor(content):
    data = [content]
    vect = cv.transform(data).toarray()
    model = pickle.load(open("Models/model.pkl", "rb"))
    result = model.predict(vect)
    if result[0] == 0:
        pred = "Negative Content"
    else:
        pred = "Positive Content"
    return pred
