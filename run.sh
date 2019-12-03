#!/bin/sh
install_dependencies() {
    pip3 install flask
    pip3 install -U scikit-learn
    python3 -m venv venv
    source venv/bin/activate
}

exp_flask() {
    export FLASK_APP=predict.py
    flask run
}

echo "Installing packages/dependencies"
install_dependencies
echo " Dependencies installed. Now Loading pretrained models"
exp_flask
