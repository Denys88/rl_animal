import gym
import numpy as np
from gym import spaces
from collections import deque
import cv2
import hyperparams as hps



def calc_rewards_v1(reward):
    if np.abs(reward - 2.5) < 0.1:
        reward = 3
    if reward > 0.1 and reward < 2.4:
        reward = 2
    if reward > 2.5:
        reward = 4
    return reward

def calc_rewards_v2(reward, vel, penalize_back = hps.BACK_MOVE_PENALTY, reward_up = hps.REWARD_RAMPS):
    if reward > 0.1:
        reward += 0.5
    if reward_up and vel[1] > 0.01:
        reward += vel[1] * hps.RAMPS_COEF
    if penalize_back and vel[2] < 0:
        reward += vel[2] * hps.BACK_MOVE_COEF

    return reward

class AnimalSkip(gym.Wrapper):
    def __init__(self, env,skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        self._skip=skip
        self.brain = env.brain
        

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        
        total_reward = None
        done = None
        frame = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self.frames_time -= 1
            vel  = obs[1]

            reward = calc_rewards_v2(reward, vel)

            frame = obs
            if total_reward is None:
                total_reward = reward
            else:
                total_reward += reward
            if done:     
                break
        
            ''' print('episode finished loose', self.frames_time)
                if(self.frames_time == -1):
                    total_reward -= 1.0
            '''
        

        return frame, total_reward, done, info

    def reset(self, config=None):
        if config == None:
            self.frames_time =  250
        else:
            self.frames_time = config.arenas[0].t
        return self.env.reset(config)

class AnimalStack(gym.Wrapper):
    def __init__(self, env, k = 2, k_vels = 2, greyscale=True):
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.k_vels = k_vels
        self.black_prob = hps.BLACK_PROBABILITY
        self.max_black_frames = hps.MAX_BLACK_FRAMES - hps.MIN_BLACK_FRAMES
        self.curr_frame = 0
        self.frames = deque([], maxlen=k)
        self.vel_info = deque([], maxlen=k_vels)
        self.pos = [0,0,0]
        self.time = 0
        self.greyscale=greyscale
        self.prev_frame = None
        shp = env.observation_space.shape
        if greyscale:
            shape = (shp[:-1] + (shp[-1] + k - 1,))
        else:
            shape = (shp[:-1] + (shp[-1] * k,))
        self.observation_space = spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)

    def reset(self, config = None):

        if config == None:
            self.time = 1.0
            self.frames_time =  250
        else:
            self.frames_time =  config.arenas[0].t
            self.time = config.arenas[0].t / 250.0

        self.pos = [0,0,0]
        frames, vel = self.env.reset(config)
        frames = np.asarray(frames * 255.0, dtype=np.uint8)
        vel = np.append(vel, [[self.time]])
        self.frames.append(frames)

        if self.greyscale:
            self.prev_frame = np.expand_dims(cv2.cvtColor(frames, cv2.COLOR_RGB2GRAY), axis=-1)
            for _ in range(self.k-1):
                self.frames.append(self.prev_frame)
        else:
            for _ in range(self.k-1):
                self.frames.append(frames)



        for _ in range(self.k_vels-1):
            self.vel_info.append([0, 0, 0, self.time])
            #self.vel_info.append(self.pos)
        self.vel_info.append(vel)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        frames = ob[0]
        vel  = ob[1]
        self.frames_time -= 1
        self.time -= 1.0 / 250.0
        self.pos += vel / 255.0
        vel = np.append(vel, [[self.time]])
        frames = np.asarray(frames * 255.0, dtype=np.uint8)
        
        if self.curr_frame == 0:
            if np.random.rand() <= self.black_prob:
                self.curr_frame = np.random.randint(1, self.max_black_frames) + hps.MIN_BLACK_FRAMES
                print('started black:', self.curr_frame)

        if self.curr_frame > 0:
            self.curr_frame = self.curr_frame - 1
            frames = frames * 0

        if self.greyscale:
            self.frames[self.k-1] = self.prev_frame
            self.prev_frame = np.expand_dims(cv2.cvtColor(frames, cv2.COLOR_RGB2GRAY), axis=-1)

        self.frames.append(frames)
        self.vel_info.append(vel)
        #self.vel_info.append(self.pos)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        stacked_frames = np.concatenate(self.frames, axis=-1)
        stacked_vels = np.concatenate(self.vel_info, axis=-1)
        
        return [stacked_frames, stacked_vels]

class ActionFlattenerVec:

    def __init__(self, n_agents):
        self.action1 = 3
        self.action2 = 3
        self.n_agents = n_agents
        if n_agents > 1:
            self.action_space = spaces.MultiDiscrete([n_agents, self.action1 * self.action2])
        else:
            self.action_space = spaces.Discrete((self.action1 * self.action2))


    def lookup_action(self, action):
        """
        Convert a scalar discrete action into a unique set of branched actions.
        :param: action: A scalar value representing one of the discrete actions.
        :return: The List containing the branched actions.
        """
        acts1 = action // self.action1
        acts2 = action % self.action1
        if self.n_agents == 1:
            return [acts1, acts2]
        res = [ [ None for y in range( 2) ] for x in range( self.n_agents  ) ]

        for i in range(0, self.n_agents):
            res[i][0] = acts1[i]
            res[i][1] = acts2[i]
        return res



class AnimalWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

        depth = 3
        n_agents = self.env.env._n_agents
        image_space_max = 1.0
        image_space_dtype = np.uint8
        camera_height = self.env.brain.camera_resolutions[0]["height"]
        camera_width = self.env.brain.camera_resolutions[0]["width"]
        if n_agents > 1:
            image_space = spaces.Box(
                0, image_space_max,
                dtype=image_space_dtype,
                shape=(n_agents, camera_height, camera_width, depth)
            )
        else:
            image_space = spaces.Box(
                0, image_space_max,
                dtype=image_space_dtype,
                shape=(camera_height, camera_width, depth)
            )
        self._flattener = ActionFlattenerVec(n_agents)
        self.observation_space = image_space
        self.action_space = self._flattener.action_space
        
    def reset(self, config = None):
        ob = self.env.reset(config)
        ob0 = np.asarray(ob[0], dtype=np.float32)
        shape = np.shape(ob0)
        if shape[0] == 1:
            ob0 = np.squeeze(ob0, axis = 0)
        return [ob0, np.asarray(ob[1], dtype=np.float32)/hps.VEC_SCALE]

    def render(self, mode='rgb_array'):
        return np.asarray(self.env.env.visual_obs * 255.0, dtype=np.uint8)

    def step(self, action):
        act = self._flattener.lookup_action(action)
        ob, reward, done, info = self.env.step(act)
        
        ob0 = np.asarray(ob[0], dtype=np.float32)
        shape = np.shape(ob0)
        if shape[0] == 1:
            ob0 = np.squeeze(ob0, axis = 0)

        return [ ob0, np.asarray(ob[1], dtype=np.float32)/hps.VEC_SCALE], np.asarray(reward), np.asarray(done, dtype=np.bool), np.asarray(info)
