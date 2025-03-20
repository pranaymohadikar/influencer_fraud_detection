import logging
import pandas as pd
import joblib
import os

logging.basicConfig(filename = 'logs/predicion_instanceMethod.log',
                    level = logging.INFO,
                    format = "%(asctime)s - %(levelname)s - %(message)s")

class Prediction:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        
    def load_model(self):
        try:
            logging.info("loading trained model")
            self.model = joblib.load(self.model_path)
            logging.info('model successfully loaded')
        except Exception as e:
            logging.error(f'error in loading model: {e}')
            print(f'error in loading model: {e}') 
            
    def predict(self, data):
           