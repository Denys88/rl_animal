import wrappers
import gym
import numpy as np
from gym import spaces
from collections import deque


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

            frame = obs
            if total_reward is None:
                total_reward = reward
            else:
                total_reward += reward
            if done:
                break

        return frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class AnimalStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        shape = (shp[:-1] + (shp[-1] * k,))
        self.observation_space = spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return np.concatenate(self.frames, axis=-1)

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
        image_space_dtype = np.float32
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
        print(self.env.brain.vector_action_space_size)
        self._flattener = ActionFlattenerVec(n_agents)
        self.observation_space = image_space
        self.action_space = self._flattener.action_space
    def reset(self):
        ob = self.env.reset()
        return np.asarray(ob[0], dtype=np.float32)

    def render(self, mode='rgb_array'):
        return np.asarray(self.env.env.visual_obs * 255.0, dtype=np.uint8)

    def step(self, action):
        act = self._flattener.lookup_action(action)
        ob, reward, done, info = self.env.step(act)
        return np.asarray(ob[0], dtype=np.float32), np.asarray(reward), np.asarray(done, dtype=np.bool), np.asarray(info)
