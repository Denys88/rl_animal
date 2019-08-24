import tensorflow as tf
import numpy as np
from tf_moving_mean_std import MovingMeanStd


class BasePlayer(object):
    def __init__(self, sess, config):
        self.config = config
        self.sess = sess
        self.env_name = self.config['ENV_NAME']

    def restore(self, fn):
        raise NotImplementedError('restore')

    def get_action(self, obs, is_determenistic = False):
        raise NotImplementedError('step')
        
    def reset(self):
        raise NotImplementedError('raise')



class PpoPlayerDiscrete(BasePlayer):
    def __init__(self, sess, config):
        BasePlayer.__init__(self, sess, config)
        self.network = config['NETWORK']
        self.obs_ph = tf.placeholder('float32', (None, 84, 84, 6 ), name = 'obs')
        self.actions_num = 9
        self.mask = [False]
        self.epoch_num = tf.Variable( tf.constant(0, shape=(), dtype=tf.float32), trainable=False)#, name = 'epochs')
        self.input_obs = self.obs_ph

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
            self.logits, _ ,_, self.action, _,self.states_ph, self.masks_ph, self.lstm_state, self.initial_state = self.network(self.run_dict, reuse=False)
            self.last_state = self.initial_state
        else:
            _ ,_, self.action,  _  = self.network(self.run_dict, reuse=False)

        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

    def get_action(self, obs, is_determenistic = False):
        #if is_determenistic:
        ret_action = self.action

        logits, action, self.last_state = self.sess.run([self.logits, ret_action, self.lstm_state], {self.obs_ph : [obs[0]], self.vec_ph : [obs[1]], self.states_ph : self.last_state, self.masks_ph : self.mask})
        if is_determenistic:
            return int(np.argmax(np.squeeze(logits)))
        else:
            return int(np.squeeze(action))

    def restore(self, fn):
        self.saver.restore(self.sess, fn)

    def reset(self):
        if self.network.is_rnn():
            self.last_state = self.initial_state