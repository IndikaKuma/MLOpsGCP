import json
import os
import pickle

import pandas as pd
from flask import jsonify


class DiabetesPredictor:
    def __init__(self):
        self.model = None

    def load_model(self):
        model_name = os.environ.get('MODEL_NAME', 'Specified environment variable is not set.')
        # Loading the saved model
        self.model = pickle.load(open(model_name, 'rb'))

    # download the model
    def download_model(self):
        model_name = os.environ.get('MODEL_NAME', 'Specified environment variable is not set.')
        # Loading the saved model
        self.model = pickle.load(open(model_name, 'rb'))
        return jsonify({'message': " the model was downloaded"}), 200

    def predict_single_record(self, prediction_input):
        if self.model is None:
            self.download_model()

        df = pd.read_json(json.dumps(prediction_input), orient='records')
        y_pred = self.model.predict(df)
        print(y_pred)
        print(y_pred[0])
        status = (y_pred[0] > 0.5)
        print(status)
        return status
