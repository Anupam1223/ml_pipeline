from abc import ABC, abstractmethod

class PostTrainingAnalysis(ABC):

    def __init__(self,**kwargs):
        pass

    @abstractmethod
    def analyze(self):
        return NotImplemented