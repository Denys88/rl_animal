import tensorflow as tf
import networks
import tensorflow_probability as tfp
tfd = tfp.distributions




class BaseModel(object):
    def is_rnn(self):
        return False


class ModelA2C(BaseModel):
    def __init__(self, network):
        self.network = network
        
    def __call__(self, dict, reuse=False):
        name = dict['name']
        inputs = dict['inputs']
        actions_num = dict['actions_num']
        prev_actions_ph = dict['prev_actions_ph']
        logits, value = self.network(name, inputs, actions_num, False, reuse)
        u = tf.random_uniform(tf.shape(logits), dtype=logits.dtype)
        # Gumbel Softmax
        action = tf.argmax(logits - tf.log(-tf.log(u)), axis=-1)
        one_hot_actions = tf.one_hot(action, actions_num)
        entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=tf.nn.softmax(logits)))

        if prev_actions_ph == None:
            neglogp = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=one_hot_actions)
            return  neglogp, value, action, entropy

        prev_neglogp = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=prev_actions_ph)
        return prev_neglogp, value, action, entropy
    
class LSTMModelA2C(BaseModel):
    def __init__(self, network):
        self.network = network

    def is_rnn(self):
        return True

    def __call__(self, dict, reuse=False):
        name = dict['name']
        inputs = dict['inputs']
        actions_num = dict['actions_num']
        prev_actions_ph = dict['prev_actions_ph']
        games_num = dict['games_num']
        batch_num = dict['batch_num']
        vels_ph = dict['vels_ph']
        logits, value, states_ph, vels_ph, masks_ph, lstm_state, initial_state = self.network(name, inputs, actions_num, games_num, batch_num, vels_ph, False, reuse)
        u = tf.random_uniform(tf.shape(logits), dtype=logits.dtype)
        action = tf.argmax(logits - tf.log(-tf.log(u)), axis=-1)
        one_hot_actions = tf.one_hot(action, actions_num)
        entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=tf.nn.softmax(logits)))

        neglogp = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=one_hot_actions)
        return   logits,neglogp, value, action, entropy, states_ph, masks_ph, lstm_state, initial_state


