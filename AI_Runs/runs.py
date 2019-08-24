import importlib.util

from animalai.envs.gym.environment import AnimalAIEnv
from animalai.envs.arena_config import ArenaConfig

    
arena_config_in = ArenaConfig('/home/trrrrr/Documents/github/ml/rl_animal/AI_Runs/configs/1-Food.yaml')

print('Resetting your agent')

env = AnimalAIEnv(
    environment_filename='/home/trrrrr/Documents/github/ml/rl_animal/AI_Runs/AI Linux.x86_64',
    seed=0,
    retro=False,
    n_arenas=1,
    worker_id=1,
    docker_training=False,
    resolution=84,
    inference=True,
    arenas_configurations=arena_config_in
)