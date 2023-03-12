import json
import os
import pickle

import pandas as pd
from flask import jsonify
from google.cloud import storage


class DiabetesPredictor:
    def __init__(self):
        self.model = None

    def load_model(self):
        model_name = os.environ.get('MODEL_NAME', 'Specified environment variable is not set.')
        # Loading the saved model
        self.model = pickle.load(open(model_name, 'rb'))

    # download the model
    def download_model(self):
        project_id = os.environ.get('PROJECT_ID', 'Specified environment variable is not set.')
        model_repo = os.environ.get('MODEL_REPO', 'Specified environment variable is not set.')
        model_name = os.environ.get('MODEL_NAME', 'Specified environment variable is not set.')
        client = storage.Client(project=project_id)
        bucket = client.get_bucket(model_repo)
        blob = bucket.blob(model_name)
        blob.download_to_filename(model_name)
        # Loading the saved model
        self.model = pickle.load(open(model_name, 'rb'))
        return jsonify({'message': " the model was downloaded"}), 200

    def predict_single_record(self, prediction_input):
        if self.model is None:
            self.download_model()

        df = pd.read_json(json.dumps(prediction_input), orient='records')
        df = df[['ntp', 'age', 'bmi', 'dbp', 'dpf', 'pgc', 'si', 'tsft']]
        y_pred = self.model.predict(df)
        print(y_pred)
        print(y_pred[0])
        status = (y_pred[0] > 0.5)
        print(status)
        return status
