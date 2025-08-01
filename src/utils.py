import os
import sys

import numpy as np
import pandas as pd
import dill

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path,obj):
    '''
    Save the given object as pickle file
    '''
    try:
        # getting directory name
        dir_path = os.path.dirname(file_path)

        # Creating Directory
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)

def evaluate_model(X_train, y_train, X_test, y_test,models,params):
    '''
    Train the model on the given model list and return r2 score of model list
    '''
    try:
        report = {}

        for i in range(len(models)):
            model = list(models.values())[i]
            param = params[list(models.keys())[i]]

            # Hyperparameter Tuning
            g_cv = GridSearchCV(model,param, cv=5, n_jobs=-1)
            g_cv.fit(X_train,y_train)

            # Training the model with best hyperparameter
            model.set_params(**g_cv.best_params_)
            model.fit(X_train,y_train)

            # Evaluatig the model
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Computing R2_score
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)

            # Save the results
            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
        
    except Exception as e:
        raise CustomException(e,sys)