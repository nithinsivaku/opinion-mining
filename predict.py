#importing libraries
import os
import numpy as np
import flask
import pickle
from flask import Flask, render_template, request, jsonify

# creating instance of the class
app=Flask(__name__)

# tell flask what url shoud trigger the function index()
@app.route('/',methods= ["GET", "POST"])
def index():
    print("world")
    if request.method == "POST":
        story =  request.get_json()
        pred = "from flask"
        if len(story['content']) != 0:
            return jsonify(pred =  pred)
        else:
            return jsonify(pred = 'Input needed'), 500
    return flask.render_template('/index.html')


# prediction function
def ValuePredictor(to_predict):
    to_predict = np.array(to_predict).reshape(1,12)
    loaded_model = pickle.load(open("model.pkl","rb"))
    result = loaded_model.predict(to_predict)
    return result[0]
