import networks
import models
import tr_helpers


animal_ai = {
    'GAMMA' : 0.99,
    'TAU' : 0.9,
    'NETWORK' : models.LSTMModelA2C(networks.animal_a2c_network_lstm2),
    'REWARD_SHAPER' : tr_helpers.DefaultRewardsShaper(scale_value = 1.0),
    'NORMALIZE_ADVANTAGE' : True,
    'LEARNING_RATE' : 5e-5,
    'NAME' : 'pong',
    'SCORE_TO_WIN' : 100500,
    'GRAD_NORM' : 0.5,
    'ENTROPY_COEF' : 0.004,
    'TRUNCATE_GRADS' : True,
    'ENV_NAME' : 'AnimalAI',
    'PPO' : True,
    'E_CLIP' : 0.1,
    'NUM_ACTORS' : 24,
    'STEPS_NUM' : 256,
    'MINIBATCH_SIZE' : 2048,
    'MINI_EPOCHS' : 2,
    'CRITIC_COEF' : 1.0,
    'CLIP_VALUE' : True,
    'LR_SCHEDULE' : 'NONE',
    'NORMALIZE_INPUT' : False,
    'SEQ_LEN' : 8,
    'MAX_EPOCHS' : 12000
}