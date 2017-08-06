from . import velocity_control_waypoint_planner
from . import waypoint_planner
from . import hregion_search
from . import center_goal
from . import random_goal
from . import center_grad
from . import choose_seven

def get(env_name):
    if env_name == "VC_Waypoints":
        return velocity_control_waypoint_planner.gameEnv
    if env_name == "Waypoints":
        return waypoint_planner.gameEnv
    if env_name == "Search":
        return hregion_search.gameEnv
    if env_name == "Center":
        return center_goal.gameEnv
    if env_name == "RandRegion":
        return random_goal.gameEnv
    if env_name == "CenterGrad":
        return center_grad.gameEnv
    if env_name == "ChooseSeven":
        return choose_seven.gameEnv
    
    raise ValueError('Unknown environment name: ' + str(env_name))
    
