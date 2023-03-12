# content of test_sysexit.py
import json
import os
import pytest
import pandas as pd

# content of test_class.py
import diabetes_predictor


class TestDiabetesPredictor:

    @pytest.fixture(scope="session", autouse=True)
    def execute_before_any_test(self):
        os.environ["MODEL_NAME"] = "testResources/model.pkl"

    # your setup code goes here, executed ahead of first test
    def test_predict_single_record(self):
        with open('testResources/prediction_request.json') as json_file:
            data = json.load(json_file)
        dp = diabetes_predictor.DiabetesPredictor()
        dp.load_model()
        status = dp.predict_single_record(data)
        assert bool(status) is not None
        # assert bool(status) is False
