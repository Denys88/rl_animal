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
        self.resolution = 84
        self.model_path = '/aaio/data/last84_10_6'
        #self.model_path = '/aaio/data/last84_9_5'
        #self.model_path = '/aaio/data/last84_8_color_b_0001'
        #self.model_path = '/aaio/data/last84_7_colorAnimalAIRay'
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
        seed=228
        np.random.seed(seed)
        tf.random.set_random_seed(seed)
        self.frames = deque([], maxlen=2)
        self.vel_info = deque([], maxlen=2) 
        self.policy.reset()
        self.first_step = True
        self.time = t / 250.0
        print('end reset agent')

    def preprocess_obs(self, brain_info):
        #v_scale = [16.0, 1.0, 16.0]
        v_scale10 = [1.0, 1.0, 16.0]
        vis = brain_info.visual_observations
        vec = brain_info.vector_observations
        vis = np.squeeze(vis)
        vec = np.squeeze(vec) / v_scale10
        vis = np.asarray(vis * 255.0, dtype=np.uint8)
        self.frames.append(vis)
    

        if self.first_step:
            self.frames.append(vis)
            self.frames.append(vis)
            self.vel_info.append([0, 0, 0, self.time])
            self.vel_info.append([0, 0, 0, self.time])
            self.first_step = False
        else:
            self.time -= 1.0 / 250.0
            vec = np.append(vec, [[self.time]])
            self.vel_info.append(vec)
            
        stacked_frames = np.concatenate(self.frames, axis=-1)
        stacked_vels = np.concatenate(self.vel_info, axis=-1)
        print(self.time)
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

        return [action // 3, action % 3]
