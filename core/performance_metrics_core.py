from abc import ABC, abstractmethod

class PerformanceMetrics(ABC):

    def __init__(self,**kwargs):
        pass

    @abstractmethod
    def return_metrics(self):
        return NotImplemented