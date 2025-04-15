from abc import ABC, abstractmethod

class Data(ABC):

    def __init__(self,**kwargs):
        pass

    @abstractmethod
    def return_data(self):
        return NotImplemented