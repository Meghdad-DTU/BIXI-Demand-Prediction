import os
import sys

sys.path.append("BIXI-Demand-Prediction/src")
from exception import CustomException
from logger import logging

import pandas as pd
from dataclasses import dataclass
from data_transformation import DataTransformation


@dataclass
class DataIngestionConfig:
    train_data_path: str= os.path.join('artifacts','train.csv')
    test_data_path: str= os.path.join('artifacts','test.csv')
    raw_data_path: str= os.path.join('artifacts','data.csv')

class DataIngestion():
    def __init__(self, train_test_ratio):
        self.ingestion_config=DataIngestionConfig()
        self.train_test_ratio = train_test_ratio
    
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or components")
        try:
            df=pd.read_csv("data/bixi_wrt_cal_15min.csv")
            logging.info('Read the dataset as dataframe')
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            
            logging.info('Train test split for time series initiated')
            ref_index = int(self.train_test_ratio*len(df))
            train_set = df.iloc[:ref_index,]
            test_set = df.iloc[ref_index:,]
            
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info("Ingestion of data is completed")
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path)
        
        
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == '__main__':
    obj=DataIngestion(0.8)
    train_data, test_data =obj.initiate_data_ingestion()
    
    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformatio(train_data, test_data)
    
            