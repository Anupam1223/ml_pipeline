from abc import ABC, abstractmethod

class DataProcessing(ABC):

    def __init__(self,**kwargs):
        pass

    @abstractmethod
    def preprocess_inputdata(self):
        return NotImplemented

    @abstractmethod
    def preprocess_targetdata(self):
        return NotImplemented