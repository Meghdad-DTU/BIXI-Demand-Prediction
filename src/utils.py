import os
import sys
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
#sys.path.append("BIXI-Demand-Prediction/src")
from src.exception import CustomException


#######################################################################
# the source of randomness can be fixed to make results reproducible
from numpy.random import seed
import itertools
#######################################################################
from keras.models import Sequential, Model
from keras.layers import Input, Dense, LSTM, Bidirectional, Conv1D, MaxPooling1D, Dropout, Flatten, BatchNormalization, concatenate, TimeDistributed
from keras.callbacks import ModelCheckpoint, EarlyStopping

#######################################################################
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
##########################################################################
import tensorflow as tf
from keras.models import save_model, load_model


class utility:
    
    def save_object(file_path, obj, h5=False):
        try:
            dir_path = os.path.dirname(file_path)
            os.makedirs(dir_path, exist_ok=True)
            if h5:
                save_model(obj, file_path)
                
            else:
                with open(file_path, 'wb') as file_obj:
                    pickle.dump(obj, file_obj)
    
        except Exception as e:
            raise CustomException(e, sys)

    def load_object(file_path, h5=False):
        try:
            if h5:
                return load_model(file_path)
            else:
                with open(file_path, 'rb') as file_obj:
                    return pickle.load(file_obj)
        
        except Exception as e:
            raise CustomException(e, sys)

        
    def split_sequence(sequence, lag, new_input=False):
        '''
        This function splits a given univariate sequence into
        multiple samples where each sample has a specified number
        of time steps and the output is a single time step.
        param new_input: If True it is used for predicting new input
        '''
        try:
            if new_input:
                X = list()
                for i in range(len(sequence)):
                    # find the end of this pattern
                    end_ix = i + history_length
                    # check if we are beyond the sequence
                    if end_ix > len(sequence):
                        break
                    # gather input and output parts of the pattern
                    seq_x = sequence[i:end_ix]
                    X.append(seq_x)
                return np.array(X)
                
            else:
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
                x, y = utility.split_sequence(dat[:,i], lag)
                X[:,:,i] = x
                Y[:,i] = y
            return X, Y
    
        except Exception as e:
            raise CustomException(e, sys)
        
    def convert_to_CNN_input(X, n_seq, n_steps):
        try:
            m, _, n = X.shape            
            return X.reshape((m, n_seq, n_steps, n))
        
        except Exception as e:
            raise CustomException(e, sys)

            
    def integer_factor(n):
        """ This function calculates integer factorization of n
        and then returns them as two multiplications
        """
        def is_prime(n):
            if n == 1:
                return False
            if n % 2 == 0:
                return False
            i = 3
            while i * i <= n:
                if n % i == 0:
                    return False
                i += 2
            return True

        def prime_factors(n):
            prime_factor_list = []
            prime_factor_list.append(1)
            for i in itertools.chain([2], itertools.count(3, 2)):
                if n <= 1:
                    break
                while n % i == 0:
                    n //= i
                    prime_factor_list.append(i)
            return prime_factor_list
        
        lst = prime_factors(n)
        lng = len(lst)
        half= int(np.round(lng/2+1))
        list1, list2 = lst[0:half], lst[half:]
        n_steps, n_sequence = np.prod(list1), np.prod(list2)
        return n_steps, n_sequence

    ######################### Deep Learning Models Archituctures ############################
    # LSTM family
    def one_LSTM(n_steps_in, n_features, n_steps_out, lstm_size, dropout, dc_size):
        model = Sequential()
        model.add(BatchNormalization(input_shape=(n_steps_in, n_features)))
        model.add(LSTM(lstm_size, dropout=dropout, recurrent_dropout=0, activation='tanh', return_sequences=True))
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dense(dc_size))
        model.add(Dropout(dropout))
        model.add(Dense(n_steps_out))
        model.compile(optimizer='adam', loss='mean_squared_error',metrics=['MSE'])
        return model

    def one_biLSTM(n_steps_in, n_features, n_steps_out, lstm_size, dropout, dc_size):
        model = Sequential()
        model.add(BatchNormalization(input_shape=(n_steps_in, n_features)))
        model.add(Bidirectional(LSTM(lstm_size, dropout=dropout, recurrent_dropout=0, activation='tanh', return_sequences=True)))
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dense(dc_size))
        model.add(Dropout(dropout))
        model.add(Dense(n_steps_out))
        model.compile(optimizer='adam', loss='mean_squared_error',metrics=['MSE'])
        return model

    
    ## CNN-LSTM family
    def TreNet_LSTM(n_steps_in, n_features, n_steps_out, filters, lstm_size, dropout, dc_size):
        n_steps, n_seq =  utility.integer_factor(n_steps_in)
        model = Sequential()
        model.add(TimeDistributed(Conv1D(filters=filters, kernel_size=1),
                          input_shape=(None, n_steps, n_features)))
        model.add(TimeDistributed(Conv1D(filters=filters, kernel_size=1, activation='relu')))
        model.add(TimeDistributed(Flatten()))
        model.add(LSTM(lstm_size, activation='tanh', recurrent_dropout=0))
        model.add(Dense(dc_size))
        model.add(Dropout(dropout))
        model.add(Dense(n_steps_out))
        model.compile(optimizer='adam', loss='mean_squared_error',metrics=['MSE'])
        return model

    def TreNet_biLSTM(n_steps_in, n_features, n_steps_out, filters, lstm_size, dropout, dc_size):
        n_steps, n_seq =  utility.integer_factor(n_steps_in)
        model = Sequential()
        model.add(TimeDistributed(Conv1D(filters=filters, kernel_size=1),
                          input_shape=(None, n_steps, n_features)))
        model.add(TimeDistributed(Conv1D(filters=filters, kernel_size=1, activation='relu')))
        model.add(TimeDistributed(Flatten()))
        model.add(Bidirectional(LSTM(lstm_size, dropout=dropout, recurrent_dropout=0, activation='tanh')))
        model.add(Dense(dc_size))
        model.add(Dropout(dropout))
        model.add(Dense(n_steps_out))
        model.compile(optimizer='adam', loss='mean_squared_error',metrics=['MSE'])
        return model
        
    
    ######################### Model Performance & Visualization ############################
    def model_loss(history,label2='Validation Loss'):
        plt.figure(figsize=(8,4))
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label=label2)
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epochs')
        plt.legend(loc='upper right')
        plt.grid(linestyle="--")
        plt.show();

    def evaluate_forecasts(actual, predicted, text = "Test", plot=True):
        """
        Evaluate prediction performance based on RMSE and MAE
        """
        RMSEs = list()
        MAEs = list()
        # calculate an RMSE score for each day
        for i in range(actual.shape[1]):
            # calculate mse
            mse = mean_squared_error(actual[:, i], predicted[:, i])
            # calculate rmse
            rmse = np.sqrt(mse)
            # store
            RMSEs.append(rmse)

            # calculate mae
            mae = mean_absolute_error(actual[:,i], predicted[:,i])
            # store
            MAEs.append(mae)

        # calculate overall RMSE and MAE
        y_true = actual.flatten()
        y_hat = predicted.flatten()

        overal_mae = mean_absolute_error(y_true, y_hat)
        overal_rmse = np.sqrt(mean_squared_error(y_true, y_hat))

        print("#### Evaluating performance metrics ####")
        print("\n===="+ text+" SET ====")
        print("MAE: {0:.3f}".format(overal_mae))
        print("RMSE: {0:.3f}".format(overal_rmse))
        print("MAEs: ", np.round(MAEs,3))
        print("RMSEs: ", np.round(RMSEs,3))

        if plot:
            plt.plot(np.arange(len(RMSEs)), RMSEs, label=True)
            plt.plot(np.arange(len(MAEs)), MAEs, label=True)
            plt.grid(linestyle="--")
            plt.xlabel("Community number")
            plt.legend(["RMSE", "MAE"])
            plt.title("Performance metrics for "+ text +" dataset")
            plt.show()

        return overal_mae, MAEs, overal_rmse, RMSEs
        
    ############################### Running Models #######################################    
class RunningDeepLearningModel:
    def __init__(self, X_train, y_train, X_test, y_test, train_val_ratio=0.8):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test 
        self.y_test = y_test          
        self.train_val_ratio = train_val_ratio
        
    
    
    def model_performance(self, Model, lstm_size, dropout, dc_size, batch_size, epoch, patience, filters, **kwargs):
        self.epoch = epoch
        self.batch_size = batch_size
        
        try:
            if Model.__name__ in ['TreNet_LSTM', 'TreNet_biLSTM']:
                CNN = True
                assert filters is not None, 'WARNING: Provide number of filters in payload!'
            elif Model.__name__ in ['one_LSTM', 'one_biLSTM']:
                CNN = False
                assert filters is None, 'WARNING: Change filters to None in payload!'
            else:
                return ('Warning: The model does not exist!')

            n_features, n_steps_out, n_steps_in = self.X_train.shape[2], self.X_train.shape[2], self.X_train.shape[1]
           
            if CNN:
                assert filters is not None, 'WARNING: Number of filters must be provided!'
                n_steps, n_seq =  utility.integer_factor(n_steps_in)
               
                # reshape from [samples, timesteps, features] into [samples, subsequences, timesteps, features]
                self.X_train = self.X_train.reshape((self.X_train.shape[0], n_seq, n_steps, n_features))
                self.X_test = self.X_test.reshape((self.X_test.shape[0], n_seq, n_steps, n_features))
                
                input_size = {'n_seq':n_seq, 'n_steps':n_steps}
                model = Model(n_steps_in, n_features, n_steps_out, filters, lstm_size, dropout, dc_size)

            else:
                input_size = {'n_seq':None, 'n_steps':None}
                model = Model(n_steps_in, n_features, n_steps_out, lstm_size, dropout, dc_size)
        
            model.summary()            
            history = model.fit(
                self.X_train,
                self.y_train,
                epochs=epoch,
                batch_size=batch_size,
                verbose=2,
                validation_split=self.train_val_ratio,
                callbacks=[EarlyStopping(monitor='val_loss', patience=patience)]
                )
            
            results=dict()
            train_set, test_set = dict(), dict()
            utility.model_loss(history)

            ## Forcasting test and training data
            y_hat_test  = model.predict(self.X_test)
            y_hat_train  = model.predict(self.X_train)

            mae_test, MAEs_test, rmse_test, RMSEs_test = utility.evaluate_forecasts(self.y_test, y_hat_test, "Test")
            mae_train, MAEs_train, rmse_train, RMSEs_train = utility.evaluate_forecasts(self.y_train, y_hat_train, "Train")
            
            train_set['mae'], train_set['MAEs'], train_set['rmse'], train_set['RMSEs'] = mae_train, MAEs_train, rmse_train, RMSEs_train
            test_set['mae'], test_set['MAEs'], test_set['rmse'], test_set['RMSEs'] = mae_train, MAEs_train, rmse_train, RMSEs_train 
            results['train_performance'], results['test_performance'] = train_set, test_set

            return Model.__name__, model, results, input_size
        
        except Exception as e:
            raise CustomException(e, sys)   
  
            
            

  