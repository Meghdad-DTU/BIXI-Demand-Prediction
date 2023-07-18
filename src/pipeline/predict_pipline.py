import os
import sys
import pandas as pd

sys.path.append("/home/paladin/Downloads/BIXI-Demand-Prediction/src/")
from exception import CustomException
from utils import utility


class PredictionPipeline:
    def __init__(self):
        pass
    
    def predict(self, features):
        try:            
            model_path = os.path.join('artifacts','Trenet_LSTM.h5')
            preprocessor_path= os.path.join('artifacts', 'preprocessor.pkl')
            input_size_path= os.path.join('artifacts', 'model_input_size.pkl')
        
            
            preprocessor = utility.load_object(file_path = preprocessor_path)
            input_size = utility.load_object(file_path = input_size_path)
            n_seq, n_steps = input_size['n_seq'], input_size['n_steps']
            
            model = utility.load_object(file_path = model_path, h5=True)
                   
            data_scaled, _ = preprocessor.transform(features)
            data_scaled = utility.convert_to_CNN_input(data_scaled, n_seq=n_seq, n_steps= n_steps)
            preds = preprocessor.inverse_transform(model.predict(data_scaled))
            
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, com1, com2, com3, com4, com5, com6):
        self.com1 = com1
        self.com2 = com2
        self.com3 = com3
        self.com4 = com4
        self.com5 = com5
        self.com6 = com6
    
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                'com1': self.com1,
                'com2': self.com2 ,
                'com3': self.com3 ,
                'com4': self.com4,
                'com5': self.com5,
                'com6': self.com6                 
            }
            
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)


                
    