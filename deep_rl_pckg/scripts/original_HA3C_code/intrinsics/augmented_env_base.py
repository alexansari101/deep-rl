

class Augmented_Env_Base():
    """Abstract class for defining how a subgoal augments an env"""

    def __init__(self, env):

        self.env = env

        self.action_space = env.action_space

        self.observation_space = lambda: None
        self.observation_space.shape = None #Must be defined in derived class

        self.m_action_space = lambda: None
        self.m_action_space.n = None #Must be defined in derived class

        self.m_observation_space = env.observation_space


    def step(self, a):
        """Returns the raw, un-augmented observations and rewards"""
        return self.env.step(a)

    def reset(self, pose_init):
        """Resets the env and returns the raw-unaugmented observation"""
        return self.env.reset(pose_init)
    
    def set_meta_actions(self, m_a):
        raise Exception("NotImplementedException")
    
    def augment_obs(self,s):
        """ Augments the raw observation with the meta_action
        """
        raise Exception("NotImplementedException")

    def intrinsic_reward(self, s, a, sp, f, m_d):
        """augments the extrinsic reward
        Returns:
        intrinsic, extrinsic, sub_agent_done"""
        raise Exception("NotImplementedException")
    

