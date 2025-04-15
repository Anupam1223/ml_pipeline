from abc import ABC, abstractmethod

class Visualization(ABC):

    def __init__(self,**kwargs):
        pass

    @abstractmethod
    def visualize(self):
        return NotImplemented