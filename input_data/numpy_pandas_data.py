from core.data_core import Data
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

class NumpyPandasData(Data):

    # Initialize:
    #***********************
    def __init__(self,config):
        self.data_path = config['data_path'] if 'data_path' in config else ""
        self.validation_split = config['validation_split'] if 'validation_split' in config else 0.1
        self.feature_names = config['feature_names'] if 'feature_names' in config else None
        self.target_names = config['target_names'] if 'target_names' in config else None
        self.n_features = config['n_features'] if 'n_features' in config else None

        self.data_is_npy = False
    #***********************

    # Load the data:
    #***********************
    def load_data(self):
        if '.csv' in self.data_path:
            return pd.read_csv(self.data_path)
        
        elif '.json' in self.data_path:
            return pd.read_json(self.data_path)
        
        elif '.feather' in self.data_path:
            return pd.read_feather(self.data_path)
        
        elif '.npy' in self.data_path:
            self.data_is_npy = True
            return np.load(self.data_path)
        else:
            print(">>> Numpy Pandas Data WARNING: Data format currently not supported <<<")
        
        # Feel free to add more data formats...
    #***********************

    # Get the training / validation data:
    #***********************
    def return_data(self):
        data = self.load_data()
        
        if self.data_is_npy == False:
           inputs = data[self.feature_names]
           targets = data[self.target_names]
           x_train, x_test, y_train, y_test = train_test_split(inputs,targets,test_size=self.validation_split)

           return {
              'training_inputs': x_train,
              'validation_inputs': x_test,
              'training_targets': y_train,
              'validation_targets': y_test
           }
        else:
            n_targets = data.shape[1] - self.n_features
            inputs = data[:,:self.n_features]
            targets = data[:,-n_targets:]

            x_train, x_test, y_train, y_test = train_test_split(inputs,targets,test_size=self.validation_split)
            return {
               'training_inputs': x_train,
               'validation_inputs': x_test,
               'training_targets': y_train,
               'validation_targets': y_test
            }
    #***********************

