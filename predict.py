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
        story = request.form.get('story')
        pred = "from flask"
        if story:
            return jsonify(pred)
        else:
            return jsonify(pred = 'Input needed')
    return flask.render_template('/index.html')
