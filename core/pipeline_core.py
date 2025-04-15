from abc import ABC, abstractmethod

class Pipeline(ABC):

    def __init__(self,**kwargs):
        pass

    @abstractmethod
    def run(self):
        return NotImplemented