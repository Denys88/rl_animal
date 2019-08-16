from animalai.envs.brain import BrainParameters
from collections import deque
import games_configurations
import players
import tensorflow as tf
import numpy as np

class Agent(object):

    def __init__(self):
        """
         Load your agent here and initialize anything needed
        """
        print('init1')
        self.resolution = 96
        self.model_path = '/aaio/data/last96_an12'
        a2c_config = games_configurations.animal_ai
        tf_config = tf.ConfigProto(
        device_count = {'GPU': 0}
        )
        print('init2')
        sess = tf.InteractiveSession()
        # Load the configuration and model using ABSOLUTE PATHS
        self.policy = players.PpoPlayerDiscrete(sess, a2c_config)
        print('init3')
        #
        self.policy.restore(self.model_path)

        print('ai loaded')

    def reset(self, t=250):
        """
        Reset is called before each episode begins
        Leave blank if nothing needs to happen there
        :param t the number of timesteps in the episode
        """
        print('start reset agent')
        self.frames = deque([], maxlen=2)
        self.vel_info = deque([], maxlen=2) 
        self.policy.reset()
        self.first_step = True
        print('end reset agent')

    def preprocess_obs(self, brain_info):
        vis = brain_info.visual_observations
        vec = brain_info.vector_observations
        vis = np.squeeze(vis)
        vec = np.squeeze(vec) / [16.0, 4.0, 16.0]

        self.frames.append(vis)
        self.vel_info.append(vec)

        if self.first_step:
            self.frames.append(vis)
            self.vel_info.append(vec)
            self.first_step = False
        

        
        stacked_frames = np.concatenate(self.frames, axis=-1)
        stacked_vels = np.concatenate(self.vel_info, axis=-1)
        return [stacked_frames, stacked_vels]

    def step(self, obs, reward, done, info):
        """
        A single step the agent should take based on the current
        :param brain_info:  a single BrainInfo containing the observations and reward for a single step for one agent
        :return:            a list of actions to execute (of size 2)
        """

        brain_info = info['brain_info']
        obs = self.preprocess_obs(brain_info)
        
        action = self.policy.get_action(obs, False)
        #return action
        return [action // 3, action % 3]
