from core.post_training_analysis_core import PostTrainingAnalysis
from utils.config_reader import ConfigReader
import torch
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

class AnupamAnalysis(PostTrainingAnalysis):

    # Inittialize:
    #**********************************
    def __init__(self,config,device):
        self.device = device
        self.cfg_reader = ConfigReader(config)
        self.feature_names = self.cfg_reader.load_setting("feature_names",None)
        # Load the anupam metrics in order to calculate the softmax loss:
        self.anupam_softmax_fn = self.cfg_reader.load_setting("anupam_softmax_fn",None)
        assert self.anupam_softmax_fn is not None,f">>> AnupamNet: ERROR. You need to provide a function for the anupam softmax <<<"

        assert self.feature_names is not None, f">>> AnupamAnalysis: ERROR. No feature names were provided. Please set 'feature_names' in your analysis config. <<<"

        #for automatic data parser
        self.particle_classes = self.cfg_reader.load_setting('particle_classes',None)
        self.prediction_labels = self.cfg_reader.load_setting('prediction_labels', None)

        # General plotting:
        self.fontsize = self.cfg_reader.load_setting("fontsize",20)

        plt.rcParams.update({'font.size':self.fontsize})
    #**********************************
    # Compute the jef objective:
    def get_anupam_softmax(self,prediction):
        return self.anupam_softmax_fn(prediction)
    #-------------------------------

    # Run the analysis:
    #**********************************
    # Process a single data frame:
    def analyze_single_dataframe(self,model,dataFrame,prediction_name,dataFrame_name,plot_dict,dataFrame_dict):
        # Get the input data:
        input_data = dataFrame[self.feature_names].values
        # print("datframe in analysis", dataFrame)
        # print("input_data", input_data)

        # Retreive model prediction:
        prediction = torch.squeeze(model.predict(torch.as_tensor(input_data,device=self.device,dtype=torch.float32))).detach().cpu().numpy()
        prediction_softmax, probabilities = self.get_anupam_softmax(prediction)
        # print("prediction",prediction_softmax)
        # print(" ")
        # particles_clases = {0:'e+', 1:'mu+', 2:'pi+', 3:'k+', 4:'p+'}
        # particles_clases = {0:'e+', 1:'e-', 2:'g', 3:'k+', 4:'k-', 5:'mu+', 6:'mu-', 7:'p+', 8:'p-', 9:'pi+', 10:'pi-'}
        vectorized_mapping = np.vectorize(self.particle_classes.get)
        labelled_prediction = vectorized_mapping(prediction_softmax)

        # prediction_labels = ['e_pred', 'm_pred', 'p_pred']
        # prediction_labels = ['pi_pred', 'k_pred', 'p_pred']
        # prediction_labels = ['k_pred', 'm_pred', 'p_pred', 'pi_pred','e_pred' ]
        # prediction_labels = ['ep_pred', 'em_pred', 'g_pred', 'kp_pred', 'km_pred', 'mup_pred', 'mum_pred', 'pp_pred', 'pm_pred', 'pip_pred', 'pim_pred']
        probabilities_df = pd.DataFrame(probabilities, columns=self.prediction_labels)
        # print("probabilities df",probabilities_df)
        
        # Register the prediction in the dataframe:
        dataFrame[prediction_name] = labelled_prediction
        dataFrame["unlabelled_prediction"] = prediction_softmax

        #----- resetting the indices to avoid null values
        probabilities_df.reset_index(drop=True, inplace=True)
        dataFrame.reset_index(drop=True, inplace=True)
        # ---------------------------------------------
        dataFrame = pd.concat([dataFrame,probabilities_df], axis=1)
        dataFrame_dict[dataFrame_name] = dataFrame
    #-----------------------------
   
    # Analyze multiple dataframes at once:
    def analyze(self,model,dataFrame_dict,prediciton_name,output_loc):
        plots = {}
        print("output loc", output_loc)

        # #++++++++++++++++++++++++
        for key in dataFrame_dict:
            if key == "validation_df":
                self.analyze_single_dataframe(model,dataFrame_dict[key],prediciton_name,key,plots,dataFrame_dict)

            if output_loc is not None:
                if key == "validation_df":
                    dataFrame_dict[key].to_csv(output_loc+'/'+key+'.csv')
        #++++++++++++++++++++++++
        # for key in dataFrame_dict:
        #     self.analyze_single_dataframe(model,dataFrame_dict[key],prediciton_name,key,plots,dataFrame_dict)

        #     if output_loc is not None:
        #         dataFrame_dict[key].to_csv(output_loc+'/'+key+'.csv')
        # #++++++++++++++++++++++++


        return plots
    #**********************************
