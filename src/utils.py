import os
import sys
import numpy as np
import pandas as pd
import pickle
sys.path.append("BIXI-Demand-Prediction/src")
from exception import CustomException


class utility:
    
    def save_object(file_path, obj):
        try:
            dir_path = os.path.dirname(file_path)
            os.makedirs(dir_path, exist_ok=True)
        
            with open(file_path, 'wb') as file_obj:
                pickle.dump(obj, file_obj)
    
        except Exception as e:
            raise CustomException(e, sys)
        
        
    def split_sequence(sequence, lag):
        '''
        This function splits a given univariate sequence into
        multiple samples where each sample has a specified number
        of time steps and the output is a single time step.
        '''
        try:
            X, y = list(), list()
            for i in range(len(sequence)):
            # find the end of this pattern
                end_ix = i + lag

            # check if we are beyond the sequence
                if end_ix > len(sequence)-1:
                    break
            # gather input and output parts of the pattern
                seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
                X.append(seq_x)
                y.append(seq_y)
            return np.array(X), np.array(y)
    
        except Exception as e:
            raise CustomException(e, sys)
    
    def convert_to_supervised(dat, lag):
        '''
        This function takes a 2D sequennce, scales the array and splits
        a given multivariate sequence into multiple samples where each sample has a specified number
        of time steps. It returns multiple time steps as well as the scaler.
        param df (DataFrame): Bike sharing demand for each community over time
        param lag (int): History length or time lag
        '''
        
        try:
            if isinstance(dat, np.ndarray):
                pass
            else:
                dat = dat.values
            
            m, n = dat.shape
            # e.g., if lag = 7, BIXI demand of past 7*15 minutes
            X = np.zeros((m-lag,lag, n))
            Y = np.zeros((m-lag,n))

            for i in range(0,n):
                x, y = utility.split_sequence(dat[:,i],lag)
                X[:,:,i] = x
                Y[:,i] = y
            return X, Y
    
        except Exception as e:
            raise CustomException(e, sys)