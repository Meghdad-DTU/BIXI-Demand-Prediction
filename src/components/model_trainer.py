import os
import sys 
from dataclasses import dataclass

#######################################################################
# the source of randomness can be fixed to make results reproducible
from numpy.random import seed
import itertools
#######################################################################
from keras.models import Sequential, Model
from keras.utils import np_utils
from keras.layers import Input, Dense, LSTM, Bidirectional, Conv1D, MaxPooling1D, Dropout, Flatten, BatchNormalization, concatenate, TimeDistributed
from keras.callbacks import ModelCheckpoint, EarlyStopping
#######################################################################
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
##########################################################################
import tensorflow as tf

sys.path.append("BIXI-Demand-Prediction/src")
from exception import CustomException
from logger import logging
from utils import utility, RunningDeepLearningModel



@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts', 'model.h5')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    def initiate_model(self, X_train_arr, y_train_arr, X_test_arr, y_test_arr, preprocessor_path):
        try:
            logging.info("Split training and test input data")
            X_train, y_train = X_train_arr, y_train_arr
            X_test, y_test = X_test_arr, y_test_arr
            
            logging.info("Creating payloads for models")
            models = {"LSTM":{'Model':utility.one_LSTM,'lag':96,'lstm_size':50,
                                 'dropout':0.4,'dc_size': 64,'batch_size': 256,
                                 'epoch' : 400, 'patience' : 10, 'filters': None}, 
                                 
                      "bi_LSTM":{ 'Model':utility.one_biLSTM,'lag':96,'lstm_size':50,
                                 'dropout':0.4,'dc_size': 64,'batch_size': 256,
                                 'epoch' : 400, 'patience' : 10, 'filters': None},
                                 
                       "TreNet_LSTM": {'Model':utility.TreNet_LSTM,'lag':96,'lstm_size':50,
                                 'dropout':0.4,'dc_size': 64,'batch_size': 256,
                                 'epoch' : 400, 'patience' : 10, 'filters': 128},
                                 
                       "TreNet_biLSTM": {'Model':utility.TreNet_biLSTM,'lag':96,'lstm_size':50,
                                 'dropout':0.4,'dc_size': 64,'batch_size': 256,
                                 'epoch' : 400, 'patience' : 10, 'filters': 128}
                     }
            
            logging.info("Initializing deep learning models")         
            model_obj = RunningDeepLearningModel(X_train, y_train, X_test, y_test)
            rmse = 1000
            for k,v in models.items():
                payload = models[k]
                model_obj.load_input(**payload)
                model, results = model_obj.model_performance()
                if results['test_performance']['rmse'] < rmse:
                    best_model = model
                    report = results
                    rmse = results['test_performance']['rmse']
            
            logging.info(f"Best model found on testing data: {best_model.__name__}")        
                
            utility.save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model,
                h5=True)    
                

        except Exception as e:
            raise CustomException(e, sys)