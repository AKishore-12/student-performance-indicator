import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    '''
    Give input required for data transformation
    '''
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):
        '''
        This function is responsible for data transforming
        Note: output feature column should not be included in Pipeline creation
        '''
        try:
            numerical_columns= ['reading_score', 'writing_score']
            categorical_columns = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            num_pipeline = Pipeline(
                steps=[
                    # Imputer - filling missing values
                    ("imputer",SimpleImputer(strategy="median")), # use medium to compute missing values
                    ("scaler",StandardScaler())
                ]
            )

            logging.info(f"Numerical features: {numerical_columns}")

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),# use mode to compute missing values
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical features: {categorical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("categorical_pipeline",cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor


        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self,train_path,test_path):
        '''
        Transform the given data and return preprocessed data and pickle file for transformation
        '''

        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining Preprocessing Object")

            preprocessing_object = self.get_data_transformer_object()

            target_column = "math_score"

            input_feature_train_df = train_data.drop(columns=[target_column],axis=1)
            target_feature_train_df = train_data[target_column]

            input_feature_test_df = test_data.drop(columns=[target_column],axis=1)
            target_feature_test_df = test_data[target_column]
            
            logging.info('Applying preprocessing on training and testing dataframe')

            input_feature_train_df_arr = preprocessing_object.fit_transform(input_feature_train_df)
            input_feature_test_df_arr = preprocessing_object.transform(input_feature_test_df)

            # Creating train and test arr
            train_arr = np.c_[
                input_feature_train_df_arr,np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_df_arr,np.array(target_feature_test_df)
            ]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_object
            )

            logging.info("Saved preprocessing object")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)