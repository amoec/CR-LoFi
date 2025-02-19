from atcenv_gym.atcenv.src.environment_objects.airspace import Airspace
from atcenv_gym.atcenv.src.environment_objects.flight import Aircraft, Flight
from atcenv_gym.atcenv.src.environment_objects.environment import Environment
from atcenv_gym.atcenv.src.observation.observation import Observation
from atcenv_gym.atcenv.src.reward.reward import Reward
from atcenv_gym.atcenv.src.scenarios.scenario import Scenario

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np

class AtcEnv(gym.Env):
    def __init__(self,
                 environment: Environment,
                 scenario: Scenario,
                 airspace: Airspace,
                 aircraft: Aircraft,
                 observation: Observation,
                 reward: Reward,
                 seed: int) -> None:
        super(AtcEnv, self).__init__()
        
        self.num_ac_state = observation.num_ac_state
        self.environment = environment  # Currently OK
        self.scenario = scenario  # Currently OK
        self.airspace_template = airspace  # Currently OK, have to implement Merging
        self.aircraft = aircraft  # Currently OK
        self.observation = observation  # Have to implement Global for transformers, 
        self.reward = reward  # Have to implement basic reward function
        
        self.Local = spaces.Dict(
            {
                "cos(drift)": spaces.Box(-1, 1, shape=(1,), dtype=np.float64),
                "sin(drift)": spaces.Box(-1, 1, shape=(1,), dtype=np.float64),
                "airspeed": spaces.Box(-1, 1, shape=(1,), dtype=np.float64),
                "x_r": spaces.Box(-np.inf, np.inf, shape=(self.num_ac_state,), dtype=np.float64),
                "y_r": spaces.Box(-np.inf, np.inf, shape=(self.num_ac_state,), dtype=np.float64),
                "vx_r": spaces.Box(-np.inf, np.inf, shape=(self.num_ac_state,), dtype=np.float64),
                "vy_r": spaces.Box(-np.inf, np.inf, shape=(self.num_ac_state,), dtype=np.float64),
                "cos(track)": spaces.Box(-np.inf, np.inf, shape=(self.num_ac_state,), dtype=np.float64),
                "sin(track)": spaces.Box(-np.inf, np.inf, shape=(self.num_ac_state,), dtype=np.float64),
                "distances": spaces.Box(-np.inf, np.inf, shape=(self.num_ac_state,), dtype=np.float64)
            }
        )
        self.observation_space = self.Local   
        self.action_space = spaces.Box(-1, 1, shape=(2,), dtype=np.float64)
        
        # Initialize random number generator
        self.seed(seed)
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def reset(self, seed=None, options=None) -> np.ndarray:
        super().reset(seed=seed)
        airspace, flights, test = self.scenario.get_scenario(self.airspace_template,self.aircraft)
        self.environment.create_environment(airspace, flights, episode=self.scenario.episode_counter)
        return self.observation.get_observation(self.environment.flights), {}
    
    def step(self, action: np.ndarray):
        observation = self.observation.get_observation(self.environment.flights)
        done = self.environment.step(action)
        new_observation = self.observation.get_observation(self.environment.flights)
        reward = self.reward.get_reward(self.environment.flights)
        
        truncated = False
        info = self.get_info()
        
        return new_observation, reward, done, truncated, info
    
    def render(self):
        pass
    
    def close(self):
        pass
    
    def store_transition(self):
        pass
            
    def get_info(self):
        return {}
    
