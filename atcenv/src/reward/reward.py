from shapely.geometry import Point, Polygon
from typing import Tuple, Optional, List
import math
import random
import numpy as np
import pickle

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import atcenv_gym.atcenv.src.functions as fn
from atcenv_gym.atcenv.src.environment_objects.flight import Flight

class Reward(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def get_reward(self, flights: List[Flight]) -> np.ndarray:
        pass

class NoReward(Reward):

    def get_reward(self, flights: List[Flight]) -> np.ndarray:
        return np.ones((len(flights),1))
    
class DefaultReward(Reward):

    def __init__(self, 
                 intrusion_weight: float = -1.,
                 drift_weight: float = -0.1):
        super().__init__()
        self.intrusion_weight = intrusion_weight
        self.drift_weight = drift_weight

    def get_reward(self, flights: List[Flight]) -> np.ndarray:
        # Reduce flight list to include only controlled flights
        self.flights_reduced = [f for f in flights if f.control]
        reward = np.zeros(len(self.flights_reduced))
        if len(self.flights_reduced) > 0:
            reward += self.get_intrusion_reward(flights)
            reward += self.get_drift_reward()
        return reward

    def get_intrusion_reward(self, flights: List[Flight]) -> np.ndarray:
        # Identify the controlled flight
        controlled_flight = next((f for f in flights if f.control), None)
    
        # Get the position of the controlled flight
        controlled_x = controlled_flight.position.x
        controlled_y = controlled_flight.position.y
        controlled_min_distance = controlled_flight.aircraft.min_distance
    
        # Calculate distances between the controlled flight and all other flights
        distances = []
        for flight in flights:
            if flight != controlled_flight:
                distance = np.sqrt((flight.position.x - controlled_x) ** 2 + (flight.position.y - controlled_y) ** 2)
                distances.append(distance)
        
        distances = np.array(distances)
        
        # Determine the number of intrusions
        intrusions = (distances < controlled_min_distance).sum()
    
        return np.array([intrusions * self.intrusion_weight])
    
    def get_drift_reward(self) -> np.ndarray:

        drift = np.array([abs(f.drift) for f in self.flights_reduced])
        return drift * self.drift_weight