from shapely.geometry import Point, Polygon
from dataclasses import dataclass, field
from typing import Tuple, Optional
import math

import CR_LoFi.atcenv.src.units as u
import CR_LoFi.atcenv.src.functions as fn
from CR_LoFi.atcenv.src.environment_objects.airspace import Airspace

@dataclass
class Aircraft:
    """ Aircraft class

    Attributes
    ___________
    min_speed: float 
        minimum speed of this aircraft type, m/s
    max_speed: float
        maximum speed of this aircraft type, m/s
    min_distance: float
        minimum horizontal separation between this aircraft and others, m

    Methods
    ___________
    None

    """

    min_speed: float
    max_speed: float
    min_distance: float

    ac_type: Optional[str] = None

@dataclass
class Flight:
    """ Flight class

    Attributes
    ___________
    aircraft: Aircraft
        Aircraft object that this flight belongs to
    control: bool
        whether the aircraft is under control of the agent
    position: shapely.geometry.Point
        current position of the aircraft in x and y, m
    target: shapely.geometry.Point
        current destination of the aircraft in x and y, m
    optimal_airspeed: float
        optimal airspeed for this flight, m/s
    flight_type: str
        type of flight of this aircraft, e.g. enroute or merging
        used for determining the waypoints
    airspeed: float
        current airspeed of the aircraft, m/s
    track: float
        current track of the aircraft, zero heading North, radians

    Properties
    ___________
    bearing: float
        bearing from the current position to the target, zero North, radians
    components: tuple(float, float)
        x and y speed components (in m/s, eastbound & northbound positive)
    distance: float
        current distance to the target, m
    drift: float
        drift angle (difference between track and bearing) to the target, radians
    prediction: shapely.geometry.Point
        predicted position of the aircraft given current velocity and lookahead time, m
    

    Methods
    ___________
    random(airspace, aircraft, tol) -> Flight object
        class method that returns a random Flight object 
    set_waypoint(self, waypoint, airspace) -> None
        updates self.target using a user specified waypoint or the airspace object

    """

    aircraft: Aircraft
    control: bool
    position: Point
    target: Point
    optimal_airspeed: float
    flight_type: str

    airspeed: float = field(init=False)
    track: float = field(init=False)

    def __post_init__(self) -> None:
        """ Initialises the track and the airspeed of the flight object """

        self.track = self.bearing
        self.airspeed = self.optimal_airspeed
    
    @property
    def bearing(self) -> float:
        """ Bearing from the current position to the target, radians, zero north """
        dx = self.target.x - self.position.x
        dy = self.target.y - self.position.y

        compass = math.atan2(dx,dy)
        return (compass+u.circle)%u.circle
    
    @property
    def components(self) -> Tuple[float, float]:
        """ X and Y speed components (in m/s, eastbound & northbound positive) """
        vx = self.airspeed * math.sin(self.track)
        vy = self.airspeed * math.cos(self.track)

        return vx, vy
    
    @property
    def distance(self) -> float:
        """ Current distance to the target (in meters) """
        return self.position.distance(self.target)
    
    @property
    def drift(self) -> float:
        """ Drift angle (difference between track and bearing) to the target """
        drift = self.bearing - self.track
        return fn.bound_angle_positive_negative_pi(drift)

    @property
    def prediction(self, dt: Optional[float] = 20) -> Point:
        """ Predicted position of the aircraft given current velocity and lookahead time, dt """
        vx, vy = self.components
        return Point([self.position.x + vx * dt, self.position.y + vy * dt])

    @classmethod
    def random(cls, airspace: Airspace, aircraft: Aircraft, tol: float = 0):
        """ Creates a random flight 

        Parameters
        __________
        airspace: Airspace
            Airspace object in which the aircraft has to spawn
        min_speed: float
            minimum speed the created aircraft should fly
        max_speed: float
            maximum speed the created aircraft should fly
        min_distance: float
            minimum distance the aircraft should have from each other
        tol: float = 0
            minimum distance between the aircraft and its target after spawning
        
        Returns
        __________
        flight: Flight
            randomly created flight object in accordance with the Airspace type
        """

        position, target, optimal_airspeed, flight_type = airspace.random_flight(aircraft.min_speed, aircraft.max_speed, tol)
        return cls(aircraft = aircraft,
                   control = False,
                   position = position, 
                   target = target,
                   optimal_airspeed = optimal_airspeed, 
                   flight_type = flight_type)
    
    def set_waypoint(self,  waypoint: Optional[Point] = None, airspace: Optional[Airspace] = None):
        """ Updates flight.target with a new waypoint
        
        Either uses a user defined waypoint, or the airspace class in combination
        with the flight object to update self.target with a new waypoint.

        Requires either an Airspace object or a waypoint, 
        with the waypoint having higher priority

        Parameters
        __________
        waypoint: Point (optional)
            new waypoint for the flight object
        airspace: Airspace (optional)
            Airspace object in which the aircraft is flying
        
        Returns
        __________
        None
        """
        if waypoint is not None:
            self.target = waypoint
        elif airspace is not None:
            self.target = airspace.get_waypoint(self.position, self.flight_type, self.target)
        else:
            raise("Method requires either a waypoint, or an airspace class to set the new waypoint")
    
