from abc import ABC, abstractmethod

class ModelWrapper(ABC):

    def __init__(self,**kwargs):
        pass

    @abstractmethod
    def build(self):
        return NotImplemented

    @abstractmethod
    def return_hyper_parameters(self):
        return NotImplemented

    @abstractmethod
    def fit(self,X,Y):
        return NotImplemented

    @abstractmethod
    def predict(self,X):
        return NotImplemented

    
