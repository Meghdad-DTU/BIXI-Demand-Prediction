import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import utility

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
        
    def get_data_transformer_object(self):
        '''
        This funnction is responsible for data transformation
        '''
        try:
            community_demand_columns = ['com1',	'com2',	'com3', 'com4', 'com5',	'com6']
            demand_pipeline = Pipeline(
                steps = [
                    ('scaler', MinMaxScaler(feature_range=(-1, 1))),
                    ('timeseries_to_supervised', FunctionTransformer(utility.convert_to_supervised, kw_args={'columns':community_demand_columns}))
                    ]
            )
            logging.info(f'Community demand features:{community_demand_columns}')
            
            preprocessor = ColumnTransformer(
                [
                'demand_pipeline', demand_pipeline, community_demand_columns    
                ]
            )
            
            return preprocessor
                
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformatio(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
                
            logging.info('Read train and test data completed')
                
            logging.info("Obtaining preprocessing object")
            preprocessing_obj=self.get_data_transformer_object()
            
            feature_columns = ['com1',	'com2',	'com3', 'com4', 'com5',	'com6']
            input_feature_train_df = train_df[feature_columns]
            input_feature_test_df = test_df[feature_columns]
                
            logging.info(f"Applying preprocessing object on both train and test dataframes")
            X_train_arr, y_train_arr = preprocessing_obj.transform(input_feature_train_df)
            X_test_arr, y_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            logging.info('Saved preprocessing object')
            utility.save_object(
                file_path= self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj)
            
            return (
                X_train_arr,
                y_train_arr,
                X_test_arr,
                y_test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            
                
        except Exception as e:
            raise CustomException(e, sys)
            
    
    