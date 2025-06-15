# Basic Import
import os
import sys
from dataclasses import dataclass

# Modelling
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")

            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            # Model Dictionary
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Nearest Neighbour": KNeighborsRegressor(),
                "Cat Boosting": CatBoostRegressor(verbose=False),
                "XGB Regressor": XGBRegressor(),
            }

            model_report: dict = evaluate_model(X_train, y_train, X_test, y_test, models)

            # # Get best model score from dictionary
            # best_model_score = max(model_report.values())

            sorted_model = dict(sorted(model_report.items(), key = lambda x: x[1], reverse=True))

            best_model_score = list(sorted_model.values())[0]
            best_model_name = list(sorted_model.keys())[0]
            
            # # Get best model from dictionary
            # best_model_name = list(models.keys())[
            #     list(models.values()).index(best_model_score)
            # ]

            best_model = models[best_model_name]

            if best_model_score < 0.65:
                raise CustomException("No Best model found")

            logging.info("Found best model")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            logging.info("Saved the best model")

            predicted = best_model.predict(X_test)

            rsquare_score = r2_score(y_test,predicted)
            return rsquare_score

        except Exception as e:
            raise CustomException(e,sys)