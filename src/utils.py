import os
import sys 
import numpy as np 
import pandas as pd
import dill as pickle
from sklearn.metrics import r2_score



from src.exception import CustomException
from src.logger import logging

def save_object(file_path:str,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys.exc_info())




def evaluate_model(X_train, y_train, X_test, y_test, models: dict):
    try:
        report = {}

        for model_name, model in models.items():

            # Train model
            model.fit(X_train, y_train)

            # Predict test data
            y_test_pred = model.predict(X_test)

            # Calculate R2 score
            test_model_score = r2_score(
                y_test, y_test_pred
            )

            report[model_name] = test_model_score

            logging.info(
                f"{model_name} initial test R2 score: "
                f"{test_model_score:.4f}"
            )

        return report

    except Exception as e:
        raise CustomException(e, sys.exc_info())



def load_object(file_path):
    try:
        with open (file_path,'rb') as file_obj:
            return pickle.load(file_obj)


    except Exception as e:
        raise CustomException(e,sys.exc_info())