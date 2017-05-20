from . import waypoint_planner
from . import hregion_search
from . import center_goal
from . import random_goal


def get(env_name):
    if env_name == "Waypoints":
        return waypoint_planner.gameEnv
    if env_name == "Search":
        return hregion_search.gameEnv
    if env_name == "Center":
        return center_goal.gameEnv
    if env_name == "RandRegion":
        return random_goal.gameEnv
    
    raise ValueError('Unknown environment name: ' + str(env_name))
    
