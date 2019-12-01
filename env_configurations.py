import networks 
import tr_helpers
import gym
import numpy as np
from hyperparams import USE_GREYSCALE_OBSES, VISUAL_FRAMES_COUNT, VEL_FRAMES_COUNT, SKIP_FRAMES, BASE_DIR



def create_animal(num_actors=1, inference = True, config=None, seed=None):
    from animalai.envs.gym.environment import AnimalAIEnv
    from animalai.envs.arena_config import ArenaConfig
    import random
    from animalai_wrapper import AnimalWrapper, AnimalStack, AnimalSkip
    env_path = 'AnimalAI'
    worker_id = random.randint(1, 60000)
    arena_config_in = ArenaConfig(BASE_DIR + '/configs/learning/stage4/3-Food Moving.yaml')

    if config is None:
        config = arena_config_in
    else: 
        config = ArenaConfig(config)
    if seed is None:
        seed = 0#random.randint(0, 100500)
        
    env = AnimalAIEnv(environment_filename=env_path,
                      worker_id=worker_id,
                      n_arenas=num_actors,
                      seed = seed,
                      arenas_configurations=config,
                      greyscale = False,
                      docker_training=False,
                      inference = inference,
                      retro=False,
                      resolution=84
                      )
    env = AnimalSkip(env, skip=SKIP_FRAMES)                  
    env = AnimalWrapper(env)
    env = AnimalStack(env,VISUAL_FRAMES_COUNT, VEL_FRAMES_COUNT, greyscale=USE_GREYSCALE_OBSES)
    return env


configurations = {
    'AnimalAI' : {
        'ENV_CREATOR' : lambda : create_animal(),
        'VECENV_TYPE' : 'ANIMAL'
    },
    'AnimalAIRay' : {
        'ENV_CREATOR' : lambda inference=False, config=None: create_animal(1, inference, config=config),
        'VECENV_TYPE' : 'RAY'
    },

}


def get_obs_and_action_spaces(name):
    env = configurations[name]['ENV_CREATOR']()
    observation_space = env.observation_space
    action_space = env.action_space
    env.close()
    return observation_space, action_space

def register(name, config):
    configurations[name] = config
