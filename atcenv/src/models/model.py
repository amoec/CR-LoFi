from shapely.geometry import Point, Polygon
from typing import Tuple, Optional, List
import math
import random
import numpy as np
import pickle

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import atcenv.src.functions as fn
from atcenv.src.environment_objects.flight import Flight

class Model(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def get_action(self, observation: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def store_transition(self, *args) -> None:
        pass

    @abstractmethod
    def set_test(self, test: bool) -> None:
        pass

    @abstractmethod
    def setup_model(self, experiment_folder: str) -> None:
        """ initializes the model for the current experiment
        
        should either load the correct weights and make a copy of them to the experiment folder
        or create a directory in the experiment folder for saving the weights during training 
        
        Parameters
        ___________
        experiment_folder: str
            location where the output weights will be saves

        Returns
        __________
        None

        """
        pass

class Straight(Model):

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        return np.zeros((len(observation),2))
    
    def store_transition(self, *args) -> None:
        pass
    
    def set_test(self, test: bool) -> None:
        pass   

    def setup_model(self, experiment_folder: str) -> None:
        pass