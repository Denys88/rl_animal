import env_configurations
import tensorflow as tf
import numpy as np
from tf_moving_mean_std import MovingMeanStd

def rescale_actions(low, high, action):
    d = (high - low) / 2.0
    m = (high + low) / 2.0
    scaled_action =  action * d + m
    return scaled_action

class BasePlayer(object):
    def __init__(self, sess, config):
        self.config = config
        self.sess = sess
        self.env_name = self.config['ENV_NAME']
        self.obs_space, self.action_space = env_configurations.get_obs_and_action_spaces(self.env_name)

    def restore(self, fn):
        raise NotImplementedError('restore')


    def create_env(self):
        return env_configurations.configurations[self.env_name]['ENV_CREATOR']()

    def get_action(self, obs, is_determenistic = False):
        raise NotImplementedError('step')
        
    def reset(self):
        raise NotImplementedError('raise')

class PpoPlayerContinuous(BasePlayer):
    def __init__(self, sess, config):
        BasePlayer.__init__(self, sess, config)
        self.network = config['NETWORK']
        self.obs_ph = tf.placeholder('float32', (None, ) + self.obs_space.shape, name = 'obs')
        self.actions_num = self.action_space.shape[0] 
        self.actions_low = self.action_space.low
        self.actions_high = self.action_space.high
        self.mask = [False]
        self.epoch_num = tf.Variable( tf.constant(0, shape=(), dtype=tf.float32), trainable=False)

        self.normalize_input = self.config['NORMALIZE_INPUT']
        if self.normalize_input:
            self.moving_mean_std = MovingMeanStd(shape = self.obs_space.shape, epsilon = 1e-5, decay = 0.99)
            self.input_obs = self.moving_mean_std.normalize(self.obs_ph, train=False)
        else:
            self.input_obs = self.obs_ph

        self.run_dict = {
            'name' : 'agent',
            'inputs' : self.input_obs,
            'batch_num' : 1,
            'games_num' : 1,
            'actions_num' : self.actions_num,
            'prev_actions_ph' : None
        }
        self.last_state = None
        if self.network.is_rnn():
            _ ,_, self.action, _, self.mu, _, self.states_ph, self.masks_ph, self.lstm_state, self.initial_state = self.network(self.run_dict, reuse=False)
            self.last_state = self.initial_state
        else:
            _ ,_, self.action, _, self.mu, _  = self.network(self.run_dict, reuse=False)

        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

    def get_action(self, obs, is_determenistic = False):
        if is_determenistic:
            ret_action = self.mu
        else:
            ret_action = self.action

        if self.network.is_rnn():
            action, self.last_state = self.sess.run([ret_action, self.lstm_state], {self.obs_ph : obs, self.states_ph : self.last_state, self.masks_ph : self.mask})
        else:
            action = self.sess.run([ret_action], {self.obs_ph : obs})
        action = np.squeeze(action)
        return  rescale_actions(self.actions_low, self.actions_high, np.clip(action, -1.0, 1.0))

    def restore(self, fn):
        self.saver.restore(self.sess, fn)

    def reset(self):
        if self.network.is_rnn():
            self.last_state = self.initial_state
        #self.mask = [True]



class PpoPlayerDiscrete(BasePlayer):
    def __init__(self, sess, config):
        BasePlayer.__init__(self, sess, config)
        self.network = config['NETWORK']
        self.obs_ph = tf.placeholder('float32', (None, ) + self.obs_space.shape, name = 'obs')
        self.actions_num = self.action_space.n
        self.mask = [False]
        self.epoch_num = tf.Variable( tf.constant(0, shape=(), dtype=tf.float32), trainable=False)#, name = 'epochs')


        self.normalize_input = self.config['NORMALIZE_INPUT']
        if self.normalize_input:
            self.moving_mean_std = MovingMeanStd(shape = self.obs_space.shape, epsilon = 1e-5, decay = 0.99)
            self.input_obs = self.moving_mean_std.normalize(self.obs_ph, train=False)
        else:
            self.input_obs = self.obs_ph
        #self.input_obs = self.preproc_images(self.input_obs)
        self.vec_ph = tf.placeholder(tf.float32, [1, 6])
        self.run_dict = {
            'name' : 'agent',
            'inputs' : self.input_obs,
            'batch_num' : 1,
            'games_num' : 1,
            'actions_num' : self.actions_num,
            'prev_actions_ph' : None,
            'vels_ph' : self.vec_ph,
        }
        self.last_state = None
        if self.network.is_rnn():
            _ ,_, self.action, _,self.states_ph, self.masks_ph, self.lstm_state, self.initial_state = self.network(self.run_dict, reuse=False)
            self.last_state = self.initial_state
        else:
            _ ,_, self.action,  _  = self.network(self.run_dict, reuse=False)

        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())



    def preproc_images(self, input):
        rgb0 = input[:,:,:,0:3]
        rgb1 = input[:,:,:,3:6]
        rgb2 = input[:,:,:,6:9]
        rgb3 = input[:,:,:,9:12] #this one is original
        
        gray0 = tf.image.rgb_to_grayscale(rgb0)
        gray1 = tf.image.rgb_to_grayscale(rgb1)
        gray2 = tf.image.rgb_to_grayscale(rgb2)
        res = tf.concat([rgb3, gray0, gray1, gray2], axis = -1)
        
        print('grey_atatat')
        print(res.get_shape().as_list())
        return res

    def get_action(self, obs, is_determenistic = False):
        #if is_determenistic:
        ret_action = self.action

        if self.network.is_rnn():
            action, self.last_state = self.sess.run([ret_action, self.lstm_state], {self.obs_ph : [obs[0]], self.vec_ph : [obs[1]], self.states_ph : self.last_state, self.masks_ph : self.mask})
        else:
            action = self.sess.run([ret_action], {self.obs_ph : [obs[0]], self.vec_ph : [obs[1]]})

        return int(np.squeeze(action))

    def restore(self, fn):
        self.saver.restore(self.sess, fn)

    def reset(self):
        if self.network.is_rnn():
            self.last_state = self.initial_state