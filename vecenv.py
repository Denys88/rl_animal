import ray
from env_configurations import configurations, create_animal
import numpy as np
import os
from hyperparams import LEARNING_DIR, MIN_TIME, MAX_TIME 



class IVecEnv(object):
    def step(self, actions):
        raise NotImplementedError 

    def reset(self):
        raise NotImplementedError 



class RayWorker:
    def __init__(self, config_name):
        self.env = configurations[config_name]['ENV_CREATOR']()
        self.obs = self.env.reset()


        self.all_tests = []

        base_dir = LEARNING_DIR
        for file in os.listdir(base_dir):
            if file.endswith(".yaml"):
                self.all_tests.append(os.path.join(base_dir, file))
    
    def step(self, action):
        next_state, reward, is_done, info = self.env.step(action)
        if is_done:
            next_state = self.reset()
        return next_state, reward, is_done, info

    def reset(self):
        # will increase brackets
        min_time = MIN_TIME
        max_time = MAX_TIME
        from animalai.envs.arena_config import ArenaConfig
        rand_test = np.random.randint(0, len(self.all_tests))
        config = ArenaConfig(self.all_tests[rand_test])
        config.arenas[0].t = np.random.randint(min_time, max_time)
        self.obs = self.env.reset(config = config)
        return self.obs


class RayVecEnv(IVecEnv):
    def __init__(self, config_name, num_actors):
        self.config_name = config_name
        self.num_actors = num_actors
        self.remote_worker = ray.remote(RayWorker)
        self.workers = [self.remote_worker.remote(self.config_name) for i in range(self.num_actors)]

    def step(self, actions):
        newobs0, newobs1, newrewards, newdones, newinfos = [], [], [], [], []
        res_obs = []
        for (action, worker) in zip(actions, self.workers):
            res_obs.append(worker.step.remote(action))
        for res in res_obs:
            cobs, crewards, cdones, cinfos = ray.get(res)
            newobs0.append(cobs[0])
            newobs1.append(cobs[1])
            newrewards.append(crewards)
            newdones.append(cdones)
            newinfos.append(cinfos)

        return [np.asarray(newobs0), newobs1], np.asarray(newrewards), np.asarray(newdones, dtype=np.bool), np.asarray(newinfos)

    def reset(self):
        obs = [worker.reset.remote() for worker in self.workers]
        obs_res = ray.get(obs)

        newobs0, newobs1 = [], []
        for obses in obs_res:
            newobs0.append(obses[0])
            newobs1.append(obses[1])

        return [np.asarray(newobs0, dtype=np.float32), np.asarray(newobs1, dtype=np.float32)]

class RayVecEnv2(IVecEnv):
    def __init__(self, config_name, num_actors):
        self.config_name = config_name
        self.num_actors = num_actors
        self.remote_worker = ray.remote(RayWorker)
        self.workers = [self.remote_worker.remote(self.config_name) for i in range(self.num_actors)]
        self.envs_in = 2

    def step(self, actions):
        newobs0, newobs1, newrewards, newdones, newinfos = None, None, [], [], []
        res_obs = []
        for i in range(self.num_actors):
            res_obs.append(self.workers[i].step.remote(actions[self.envs_in * i : self.envs_in * (i+1)]))
        for res in res_obs:
            cobs, crewards, cdones, cinfos = ray.get(res)
            if newobs0 is None:
                newobs0 = cobs[0]
                newobs1 = cobs[1]
            else:
                newobs0 = np.vstack([newobs0, cobs[0]])
                newobs1 = np.vstack([newobs1, cobs[1]])
            newrewards.append(crewards)
            newdones.append(cdones)
            newinfos.append(cinfos)

        return [np.asarray(newobs0), newobs1], np.asarray(newrewards).flatten(), np.asarray(newdones, dtype=np.bool).flatten(), np.asarray(newinfos)

    def reset(self):
        obs = [worker.reset.remote() for worker in self.workers]
        obs_res = ray.get(obs)

        newobs0, newobs1 = None, None
        for cobs in obs_res:
            if newobs0 is None:
                newobs0 = cobs[0]
                newobs1 = cobs[1]
            else:
                newobs0 = np.vstack([newobs0, cobs[0]])
                newobs1 = np.vstack([newobs1, cobs[1]])

        return [np.asarray(newobs0, dtype=np.float32), np.asarray(newobs1, dtype=np.float32)]


    

def create_vec_env(config_name, num_actors):
    if configurations[config_name]['VECENV_TYPE'] == 'RAY':
        return RayVecEnv(config_name, num_actors)
    if configurations[config_name]['VECENV_TYPE'] == 'ANIMAL':
        return create_animal(num_actors, inference=False)