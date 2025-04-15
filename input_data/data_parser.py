from core.data_core import Data
from utils.config_reader import ConfigReader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import pandas as pd
import torch
import uproot
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class AnupamDataParser(Data):

    # Initialize:
    #***********************
    def __init__(self,config,device):
        # Get the config reader:
        self.cfg_reader = ConfigReader(config)
        self.device = device

        # General data info:
        self.seed = self.cfg_reader.load_setting("seed",None)
        if self.seed is not None:
           np.random.seed(self.seed)
        self.data_path = self.cfg_reader.load_setting("data_path","")
        self.validation_split = self.cfg_reader.load_setting("validation_split",0.3)
        self.feature_names = self.cfg_reader.load_setting('feature_names',None)
        self.target_names = self.cfg_reader.load_setting('target_names',None)
        # self.range_variable_names = self.cfg_reader.load_setting('range_variable_names',None)
        # print(self.range_variable_names)

        #for automatic label mapping
        self.label_mapping = self.cfg_reader.load_setting('label_mapping',None)

        # Weights for each bkg. or sig. decay mode:
        self.target_luminosity = self.cfg_reader.load_setting('target_luminosity',1000)
        self.decay_branches = self.cfg_reader.load_setting('decay_branches',None)
        self.luminosity_per_branch = self.cfg_reader.load_setting('luminosity_per_branch',None)
        self.luminosity_factor = self.cfg_reader.load_setting('luminosity_factor',1.0)
        self.n_decays = len(self.decay_branches)

        assert self.feature_names is not None, f">>> AnupamDataParser: ERROR. No feature names have been provided. Please provide a list of feature names. <<<"
        assert self.target_names is not None, f">>> AnupamDataParser: ERROR. No target names have been provided. Please provide a list of target names.  <<<"

        self.n_features = len(self.feature_names)

        # Name training / validation dataframe:
        self.training_df_name = self.cfg_reader.load_setting("training_df_name","training_df")
        self.validation_df_name = self.cfg_reader.load_setting("validation_df_name","validation_df")
        # Create a dictionary to store the training / validation data frames:
        self.df_dict = {}

        self.data_is_npy = False
    #***********************

    # Load the data:
    #***********************
    # Read in a single file:
    def read_single_file(self,data_loc):
        if '.csv' in data_loc:
            return pd.read_csv(data_loc)
        
        elif '.json' in data_loc:
            return pd.read_json(data_loc)
        
        elif '.feather' in data_loc:
            return pd.read_feather(data_loc)
        
        elif '.npy' in data_loc:
            self.data_is_npy = True
            return np.load(data_loc)

        elif '.root' in data_loc:
            with uproot.open(data_loc) as file:
                tree = file['trk_pid']
                
                # ------------ applying cut --------------------
                df = tree.arrays(library="pd")
                
                filtered_df = df[(df["trk_e_p"] > 0.25) & (df["trk_e_p"] < 1.25) & ((df["bcal_e_o_p"] > 0) | (df["fcal_e_o_p"] > 0)) & (df["trk_nb"] == 1)]

                return filtered_df
                # ------------------- ends applying cut ---------
                #for trk_e_p range
                #0.25 to 1 GeV/c
                #1 to 2GeV/c
                #2 to 3GeV/c
                #3 to 5GeV/c
                # return tree.arrays(library='pd')
        else:
            print(">>> AnupamDataParser WARNING: Data format currently not supported <<<")
        
        # Feel free to add more data formats...

    #-----------------------------------
    
    # Load the data:
    def load_data(self,data_loc):
        # Check if data object is a string, i.e. a single file:
        if isinstance(data_loc,str):
            return self.read_single_file(data_loc)
        
        # Check if the data is an actual list:
        if isinstance(data_loc,list):
            data_collection = []
            #+++++++++++++++++++++
            for loc in data_loc:
                data_collection.append(self.read_single_file(loc))
            #+++++++++++++++++++++

            if self.data_is_npy:
                return np.concatenate(data_collection,axis=0)
            
            return pd.concat(data_collection,axis=0)
    #***********************

    # Data pre-processing:
    #***********************

    # Add weights to the data frame:
    def add_weights_to_df(self, dataFrame):
        # Initialize weights with random values
        # Using a uniform distribution in the range [0, 1)
        np.random.seed(42)  # Seed for reproducibility (optional)
        dataFrame['weights'] = np.random.uniform(0, 1, size=len(dataFrame))
        
        # Normalize weights to ensure they sum to 1 (optional, depending on use case)
        total_weight = dataFrame['weights'].sum()
        if total_weight > 0:
            dataFrame['weights'] /= total_weight
        
        dataFrame['weights'] = dataFrame['weights'].astype('single')
        #-------------------------------

    # Now put it all together:
    def run_df_preprocessing(self,dataFrame):
        # Add the weights:
        self.add_weights_to_df(dataFrame)
    #***********************

    # feature normalization
    def normalize_features(self, data_frame, feature_names, method='standard'):
        """
        Normalize features in the given DataFrame.
        
        Parameters:
        - data_frame (pd.DataFrame): The input DataFrame containing the features.
        - feature_names (list): List of column names (features) to normalize.
        - method (str): The normalization method ('minmax' or 'standard').
        
        Returns:
        - pd.DataFrame: DataFrame with normalized features.
        """
        if method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'standard':
            scaler = StandardScaler()
        else:
            raise ValueError("Invalid method. Use 'minmax' or 'standard'.")
        
        # Only normalize the specified features
        data_frame[feature_names] = scaler.fit_transform(data_frame[feature_names])
        return data_frame

    # Get the training / validation data:
    #***********************
    # Get inputs, targets and sigmas from one dataframe:
    def get_xys(self,dataFrame):
        # Run preprocessing:
        self.run_df_preprocessing(dataFrame)
        # And extend the targets:
        if 'weights' not in self.target_names:
           self.target_names.append('weights')
        
        dataFrame = self.normalize_features(dataFrame,self.feature_names)
        inputs = dataFrame[self.feature_names]
        targets = dataFrame[self.target_names]

        # return inputs,targets
        return inputs,targets
    #----------------------------------------

    def convert_to_one_vs_all(self, dataframe):
        # Initialize the LabelBinarizer
        lb = LabelBinarizer()
        # Ensure the 'label' column is of type string (if it's not already)
        dataframe['s_label'] = dataframe['s_label'].astype(str)

        # Fit and transform the labels
        one_vs_all_labels = lb.fit_transform(dataframe['s_label'])
        # print("dataframe", dataframe['s_label'])
        # print("oneVsall", one_vs_all_labels)

        # The classes in the order they appear in the binary matrix
        classes = lb.classes_

        # Add these one-vs-all labels back to the DataFrame for inspection
        for i, class_label in enumerate(classes):
            dataframe[class_label] = one_vs_all_labels[:, i]

        # Display the DataFrame with one-vs-all labels
        # print("new dataframe",dataframe)
        # print("one vs all", one_vs_all_labels)

        # Optional: If you only need the binary matrix without the original labels:
        return one_vs_all_labels

    # Return the data:
    def return_data(self):
        self.data = self.load_data(self.data_path)
        df_train = None
        df_test = None
        
        if self.data_is_npy == False:
           # Devide the data frame into training / validation data:
           df_train, df_test = train_test_split(self.data,test_size=self.validation_split)

           x_train, y_train = self.get_xys(df_train)
           x_test, y_test = self.get_xys(df_test)

           # Register dataframes:
           self.df_dict[self.training_df_name] = df_train
           self.df_dict[self.validation_df_name] = df_test

           y_train_df = pd.DataFrame(y_train['s_label'])
           y_test_df = pd.DataFrame(y_test['s_label'])
           
           w_train_df = pd.DataFrame(y_train['weights'])
           w_test_df =  pd.DataFrame(y_test['weights'])

           # Filter out unwanted labels --------------------------------------------------
           labels_to_exclude = ['pi+']
           y_train_df = y_train_df[~y_train_df['s_label'].isin(labels_to_exclude)]
           y_test_df = y_test_df[~y_test_df['s_label'].isin(labels_to_exclude)]

            # Ensure corresponding inputs and weights are also filtered
           x_train = x_train.loc[y_train_df.index]
           x_test = x_test.loc[y_test_df.index]
           w_train_df = w_train_df.loc[y_train_df.index]
           w_test_df = w_test_df.loc[y_test_df.index]
           # ---------Filter out unwanted ends----------------------------------------------------------------------

        # this process is for converting label to its corresponding numeric values
           y_train_df_numeric = y_train_df.applymap(lambda x: self.label_mapping[x])
           y_test_df_numeric = y_test_df.applymap(lambda x: self.label_mapping[x])
        # -----------------------------------------------------------------------
           
           return {
              'training_inputs': torch.as_tensor(x_train.values,device=self.device, dtype=torch.float32),
              'validation_inputs': torch.as_tensor(x_test.values,device=self.device, dtype=torch.float32),
              'training_targets': torch.as_tensor(y_train_df_numeric.values,device=self.device, dtype=torch.float32),
              'validation_targets': torch.as_tensor(y_test_df_numeric.values,device=self.device, dtype=torch.float32),
              'weights_train': torch.as_tensor(w_train_df.values,device=self.device, dtype=torch.float32),
              'weights_test': torch.as_tensor(w_test_df.values,device=self.device, dtype=torch.float32),
           }
        else:
            n_targets = self.data.shape[1] - self.n_features
            inputs = self.data[:,:self.n_features]
            targets = self.data[:,-n_targets:]

            x_train, x_test, y_train, y_test = train_test_split(inputs,targets,test_size=self.validation_split)
            return {
               'training_inputs': torch.as_tensor(x_train,device=self.device),
               'validation_inputs': torch.as_tensor(x_test,device=self.device),
               'training_targets': torch.as_tensor(y_train,device=self.device),
               'validation_targets': torch.as_tensor(y_test,device=self.device)
            }
    #***********************

