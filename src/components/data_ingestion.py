import os
import sys
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation

# dataclass - structured way to store data
@dataclass
class DataIngestionConfig:
    '''
        Configuring the path for storing the data.
    '''
    train_data_path: str = os.path.join("artifacts","train.csv")
    test_data_path: str = os.path.join("artifacts","test.csv")
    raw_data_path: str = os.path.join("artifacts","raw.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        '''
        Read the data from database
        '''
        logging.info("Entered the data ingestion method or component")
        try:
            # Reading the data(we can read from different sources)
            df = pd.read_csv("src/notebook/data/stud.csv")
            logging.info("Read the dataset as Data frame")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            # Train-Test Split
            logging.info("Train Test Split initiated")
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)

            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of data is completed")

            return (self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)
    
        except Exception as e:
            raise CustomException(e,sys)

if __name__ == "__main__":
    # Data Ingestion
    obj = DataIngestion()
    train_path, test_path = obj.initiate_data_ingestion()

    # Data Transformation
    data_transforamation = DataTransformation()
    data_transforamation.initiate_data_transformation(train_path,test_path)
