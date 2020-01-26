



'''
Directory with test levels,
For first submitted network with score 42.66 I used stage3
For second I used stage3 and than stage4
'''
BASE_DIR = '/home/trrrrr/Documents/github/ml/animal_master/rl_animal'
LEARNING_DIR =  BASE_DIR + '/configs/learning/competition_configurations/'

'''
Minimum and maximum steps in enviroment 
'''
MIN_TIME = 200
MAX_TIME = 1100

'''
Scale of the vel inpuits
'''
VEC_SCALE = [1.0, 1.0, 16.0]


'''
Probability to generate black frames.
Tested every frame for every env. Good value is in [0.0001,0.003]
'''
BLACK_PROBABILITY = 0.0000

'''
When we generating black frames we decidin how many of them to generate.
'''
MIN_BLACK_FRAMES = 3
MAX_BLACK_FRAMES = 9

'''
Will we add penalty for moving back
'''
BACK_MOVE_PENALTY = True

'''
Reward = velocity * BACK_MOVE_COEF if velocity < 0
'''
BACK_MOVE_COEF = 1.0 / 1000.0

'''
Will we add reward for moving up
'''
REWARD_RAMPS = True

'''
Reward = vertical velocity * BACK_MOVE_COEF if velocity > 0
'''
RAMPS_COEF = 1.0 / 100.0

'''
If false then all frames are rgb. If true frist frame is RGB and all other are greyscale
'''
USE_GREYSCALE_OBSES = False

'''
number of visual frames in input
'''
VISUAL_FRAMES_COUNT = 2

'''
number of velocity frames in input
'''
VEL_FRAMES_COUNT = 2

SKIP_FRAMES = 1