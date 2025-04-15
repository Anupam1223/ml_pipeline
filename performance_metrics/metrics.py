from mlflow_jlab.core.performance_metrics_core import PerformanceMetrics
from mlflow_jlab.utils.config_reader import ConfigReader
# from scipy.special import softmax
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class AnupamMetrics(PerformanceMetrics):

    # Initialize:
    #*******************************
    def __init__(self,config,device):
        self.device = device
        self.cfg_reader = ConfigReader(config)
        
        # Parameters for the sensitivity calculation:
        self.a = self.cfg_reader.load_setting("a",3.0)
        self.b = self.cfg_reader.load_setting("b",1.28155)
        self.scaling = self.cfg_reader.load_setting("scaling",1.0)

        self.n_mass_hypotheses = self.cfg_reader.load_setting("n_mass_hypotheses",0)

        assert self.n_mass_hypotheses >0, f">>> AnupamMetric: ERROR! You did not provide a positive number of mass hypotheses: n_mass_hypotheses = {self.n_mass_hypotheses} <<<"
    #*******************************
    
    # Compute the loss:
    def compute_anupam_loss(self,network_response,target_value):
        criterion = nn.CrossEntropyLoss()

        # Calculate the loss
        loss = criterion(network_response, target_value)
        return loss
    #*******************************
    # def softmax(self,x):
    #     print("prediction from F softmax",x)
    #     e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    #     return e_x / e_x.sum(axis=1, keepdims=True)

    def compute_softmax(self,prediction):
        # Apply softmax to get probabilities
        # probabilities = self.softmax(prediction)
        probabilities = torch.tensor(prediction)
        probabilities = F.softmax(probabilities,dim=1)
        
        # Get the class with the highest probability for each data row
        predicted_classes = torch.argmax(probabilities, dim=1)

        # Detach from graph and move to CPU, then convert to numpy array
        prediction = predicted_classes.detach().cpu().numpy()

        # Ensure the result is 1D if it isn't already
        if len(prediction.shape) != 1:
            prediction = prediction.reshape(-1)

        return prediction, probabilities

    # Make the metrics available:
    #*******************************
    def return_metrics(self):
        metrics_dict = {
            'anupam_loss': self.compute_anupam_loss,
            'softmax_loss': self.compute_softmax
        }

        return metrics_dict
    #*******************************

