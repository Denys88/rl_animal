import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
tfd = tfp.distributions

def sample_noise(shape, mean = 0.0, std = 1.0):
    noise = tf.random_normal(shape, mean = mean, stddev = std)
    return noise

# Added by Andrew Liao
# for NoisyNet-DQN (using Factorised Gaussian noise)
# modified from ```dense``` function
def noisy_dense(inputs, units, name, bias=True, activation=tf.identity, mean = 0.0, std = 1.0):

    # the function used in eq.7,8
    def f(x):
        return tf.multiply(tf.sign(x), tf.pow(tf.abs(x), 0.5))
    # Initializer of \mu and \sigma 
    mu_init = tf.random_uniform_initializer(minval=-1*1/np.power(inputs.get_shape().as_list()[1], 0.5),     
                                                maxval=1*1/np.power(inputs.get_shape().as_list()[1], 0.5))
    sigma_init = tf.constant_initializer(0.4/np.power(inputs.get_shape().as_list()[1], 0.5))
    # Sample noise from gaussian
    p = sample_noise([inputs.get_shape().as_list()[1], 1], mean = 0.0, std = 1.0)
    q = sample_noise([1, units], mean = 0.0, std = 1.0)
    f_p = f(p); f_q = f(q)
    w_epsilon = f_p*f_q; b_epsilon = tf.squeeze(f_q)

    # w = w_mu + w_sigma*w_epsilon
    w_mu = tf.get_variable(name + "/w_mu", [inputs.get_shape()[1], units], initializer=mu_init)
    w_sigma = tf.get_variable(name + "/w_sigma", [inputs.get_shape()[1], units], initializer=sigma_init)
    w = w_mu + tf.multiply(w_sigma, w_epsilon)
    ret = tf.matmul(inputs, w)
    if bias:
        # b = b_mu + b_sigma*b_epsilon
        b_mu = tf.get_variable(name + "/b_mu", [units], initializer=mu_init)
        b_sigma = tf.get_variable(name + "/b_sigma", [units], initializer=sigma_init)
        b = b_mu + tf.multiply(b_sigma, b_epsilon)
        return activation(ret + b)
    else:
        return activation(ret)

def batch_to_seq(h, nbatch, nsteps, flat=False):
    if flat:
        h = tf.reshape(h, [nbatch, nsteps])
    else:
        h = tf.reshape(h, [nbatch, nsteps, -1])
    return [tf.squeeze(v, [1]) for v in tf.split(axis=1, num_or_size_splits=nsteps, value=h)]

def seq_to_batch(h, flat = False):
    shape = h[0].get_shape().as_list()
    if not flat:
        assert(len(shape) > 1)
        nh = h[0].get_shape()[-1].value
        return tf.reshape(tf.concat(axis=1, values=h), [-1, nh])
    else:
        return tf.reshape(tf.stack(values=h, axis=1), [-1])

def ortho_init(scale=1.0):
    def _ortho_init(shape, dtype, partition_info=None):
        #lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4: # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)
    return _ortho_init

def lstm(xs, ms, s, scope, nh,  nin):
    with tf.variable_scope(scope):
        wx = tf.get_variable("wx", [nin, nh*4], initializer=ortho_init(), dtype=tf.float32 )
        wh = tf.get_variable("wh", [nh, nh*4], initializer=ortho_init() )
        b = tf.get_variable("b", [nh*4], initializer=tf.constant_initializer(0.0))

    c, h = tf.split(axis=1, num_or_size_splits=2, value=s)

    for idx, (x, m) in enumerate(zip(xs, ms)):
        c = c*(1-m)
        h = h*(1-m)
        z = tf.matmul(x, wx) + tf.matmul(h, wh) + b
        i, f, o, u = tf.split(axis=1, num_or_size_splits=4, value=z)
        i = tf.nn.sigmoid(i)
        f = tf.nn.sigmoid(f)
        o = tf.nn.sigmoid(o)
        u = tf.tanh(u)
        c = f*c + i*u
        h = o*tf.tanh(c)
        xs[idx] = h
    s = tf.concat(axis=1, values=[c, h])
    return xs, s

def _ln(x, g, b, e=1e-5, axes=[1]):
    u, s = tf.nn.moments(x, axes=axes, keep_dims=True)
    x = (x-u)/tf.sqrt(s+e)
    x = x*g+b
    return x

def lnlstm(xs, ms, s, scope, nh, nin):
    with tf.variable_scope(scope):
        wx = tf.get_variable("wx", [nin, nh*4], initializer=ortho_init())
        gx = tf.get_variable("gx", [nh*4], initializer=tf.constant_initializer(1.0))
        bx = tf.get_variable("bx", [nh*4], initializer=tf.constant_initializer(0.0))

        wh = tf.get_variable("wh", [nh, nh*4], initializer=ortho_init())
        gh = tf.get_variable("gh", [nh*4], initializer=tf.constant_initializer(1.0))
        bh = tf.get_variable("bh", [nh*4], initializer=tf.constant_initializer(0.0))

        b = tf.get_variable("b", [nh*4], initializer=tf.constant_initializer(0.0))

        gc = tf.get_variable("gc", [nh], initializer=tf.constant_initializer(1.0))
        bc = tf.get_variable("bc", [nh], initializer=tf.constant_initializer(0.0))

    c, h = tf.split(axis=1, num_or_size_splits=2, value=s)
    tk = 0
    for idx, (x, m) in enumerate(zip(xs, ms)):
        print(tk)
        tk = tk + 1
        c = c*(1-m)
        h = h*(1-m)
        z = _ln(tf.matmul(x, wx), gx, bx) + _ln(tf.matmul(h, wh), gh, bh) + b
        i, f, o, u = tf.split(axis=1, num_or_size_splits=4, value=z)
        i = tf.nn.sigmoid(i)
        f = tf.nn.sigmoid(f)
        o = tf.nn.sigmoid(o)
        u = tf.tanh(u)
        c = f*c + i*u
        h = o*tf.tanh(_ln(c, gc, bc))
        xs[idx] = h
    s = tf.concat(axis=1, values=[c, h])
    return xs, s

'''
used lstm from openai baseline as the most convenient way to work with dones.
TODO: try to use more efficient tensorflow way
'''
def openai_lstm(name, inputs, states_ph, dones_ph, units, env_num, batch_num, layer_norm=True):
    nbatch = batch_num
    nsteps = nbatch // env_num
    dones_ph = tf.to_float(dones_ph)
    inputs_seq = batch_to_seq(inputs, env_num, nsteps)
    dones_seq = batch_to_seq(dones_ph, env_num, nsteps)
    nin = inputs.get_shape()[1].value
    with tf.variable_scope(name):
        if layer_norm:
            hidden_seq, final_state = lnlstm(inputs_seq, dones_seq, states_ph, scope='lnlstm', nin=nin, nh=units)
        else:
            hidden_seq, final_state = lstm(inputs_seq, dones_seq, states_ph, scope='lstm', nin=nin, nh=units)

    hidden = seq_to_batch(hidden_seq)
    initial_state = np.zeros(states_ph.shape.as_list(), dtype=float)
    return [hidden, final_state, initial_state]


def animal_conv_net(inputs):
    NUM_FILTERS_1 = 8
    NUM_FILTERS_2 = 16
    NUM_FILTERS_3 = 32
    NUM_FILTERS_4 = 64
    NUM_FILTERS_5 = 512
    conv1 = tf.layers.conv2d(inputs=inputs,
                             filters=NUM_FILTERS_1,
                             kernel_size=[5, 5],
                             strides=(4, 4),  
                             activation=tf.nn.elu)
    conv2 = tf.layers.conv2d(inputs=conv1,
                             filters=NUM_FILTERS_2,
                             kernel_size=[3, 3],
                             strides=(2, 2),         
                             activation=tf.nn.elu)
    conv3 = tf.layers.conv2d(inputs=conv2,
                             filters=NUM_FILTERS_3,
                             kernel_size=[3, 3],
                             strides=(2, 2),         
                             activation=tf.nn.elu)
    conv4 = tf.layers.conv2d(inputs=conv3,
                             filters=NUM_FILTERS_4,
                             kernel_size=[3, 3],
                             strides=(2, 2),                           
                             activation=tf.nn.elu)
    #conv5 = tf.layers.conv2d(inputs=conv4,
    #                         filters=NUM_FILTERS_5,
    #                         kernel_size=[3, 3],
    #                         strides=(1, 1),                           
    #                         activation=tf.nn.elu)
    return conv4
  

def residual_block(inputs, name):
    depth = inputs.get_shape()[-1].value
    out = tf.nn.relu(inputs)
    out = tf.layers.conv2d(out, depth, 3, padding='same', name='res1_' + name, activation=tf.nn.relu)
    out = tf.layers.conv2d(out, depth, 3, padding='same', name='res2_' + name)
    return out + inputs


def residual_block_fixup(inputs, num_layers, name):
    depth = inputs.get_shape()[-1].value
    stddev = np.sqrt(2.0/(inputs.get_shape()[1].value * inputs.get_shape()[2].value * inputs.get_shape()[3].value * num_layers))

    bias0 = tf.get_variable(name + "/bias0", [], initializer=tf.zeros_initializer())
    bias1 = tf.get_variable(name + "/bias1", [], initializer=tf.zeros_initializer())
    bias2 = tf.get_variable(name + "/bias2", [], initializer=tf.zeros_initializer())
    bias3 = tf.get_variable(name + "/bias3", [], initializer=tf.zeros_initializer())

    multiplier = tf.get_variable(name + "/multiplier", [], initializer=tf.ones_initializer())
    out = tf.nn.relu(inputs)
    out = out + bias0
    out = tf.layers.conv2d(out, depth, 3, padding='same', name='res1/' + name, use_bias = False, kernel_initializer=tf.truncated_normal_initializer(stddev=stddev))
    out = out + bias1
    out = tf.nn.relu(out)
    out = out + bias2
    out = tf.layers.conv2d(out, depth, 3, padding='same', name='res2/' + name, use_bias = False, kernel_initializer=tf.zeros_initializer())
    out = out * multiplier + bias3

    return out + inputs

def animal_conv_residual(inputs, depths = [16,32,32,64]):
    
    out = inputs
    i = 0
    for depth in depths:
        i = i + 1
        out = tf.layers.conv2d(out, depth, 3, padding='same', name='layer_' + str(i))
        out = tf.layers.max_pooling2d(out, pool_size=3, strides=2, padding='same')
        out = residual_block(out, name='rb1' + str(i))
        out = residual_block(out, name='rb2' + str(i))
        print('shapes of layer_' + str(i), str(out.get_shape().as_list()))
    out = tf.nn.relu(out)
    return out


def animal_conv_residual_fixup(inputs, depths = [16,32,32,64]):
    out = inputs
    i = 0
    for depth in depths:
        i = i + 1
        out = tf.layers.conv2d(out, depth, 3, padding='same', name='layer_' + str(i))
        out = tf.layers.max_pooling2d(out, pool_size=3, strides=2, padding='same')
        out = residual_block_fixup(out, i, name='rb1' + str(i))
        out = residual_block_fixup(out, i, name='rb2' + str(i))
        print('shapes of layer_' + str(i), str(out.get_shape().as_list()))
    out = tf.nn.relu(out)
    return out

def animal_a2c_network(name, inputs, actions_num, continuous=False, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        NUM_HIDDEN_NODES = 512

        conv3 = animal_conv_net(inputs)
        flatten = tf.contrib.layers.flatten(inputs = conv3)
        hidden = tf.layers.dense(inputs=flatten, units=NUM_HIDDEN_NODES, activation=tf.nn.elu)
  
        value = tf.layers.dense(inputs=hidden, units=1, activation=None)
        if continuous:
            mu = tf.layers.dense(inputs=hidden, units=actions_num, activation=tf.nn.tanh)
            var = tf.layers.dense(inputs=hidden, units=actions_num, activation=tf.nn.softplus)
            return mu, var, value
        else:
            logits = tf.layers.dense(inputs=hidden, units=actions_num, activation=None)
            return logits, value

def animal_a2c_network_lstm(name, inputs, actions_num, env_num, batch_num, vels_ph, continuous=False, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        NUM_HIDDEN_NODES = 256
        LSTM_UNITS = 256
        conv3 = animal_conv_residual(inputs)
        flatten = tf.contrib.layers.flatten(inputs = conv3)

        hidden0 = tf.layers.dense(inputs=vels_ph, units=32, activation=tf.nn.elu)
        hidden1 = tf.layers.dense(inputs=flatten, units=NUM_HIDDEN_NODES, activation=tf.nn.elu)

        
        dones_ph = tf.placeholder(tf.bool, [batch_num])
        states_ph = tf.placeholder(tf.float32, [env_num, 2*LSTM_UNITS])
        hidden2 = tf.concat([hidden0, hidden1], axis=-1)
        lstm_out, lstm_state, initial_state = openai_lstm('lstm_ac', hidden2, dones_ph=dones_ph, states_ph=states_ph, units=LSTM_UNITS, env_num=env_num, batch_num=batch_num)
        value = tf.layers.dense(inputs=lstm_out, units=1, activation=None)
        if continuous:
            mu = tf.layers.dense(inputs=lstm_out, units=actions_num, activation=tf.nn.tanh)
            var = tf.layers.dense(inputs=lstm_out, units=actions_num, activation=tf.nn.softplus)
            return mu, var, value, states_ph, vels_ph, dones_ph, lstm_state, initial_state
        else:
            logits = tf.layers.dense(inputs=lstm_out, units=actions_num, activation=None)
            return logits, value, states_ph, vels_ph, dones_ph, lstm_state, initial_state


def animal_a2c_network_lstm2(name, inputs, actions_num, env_num, batch_num, vels_ph, continuous=False, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        NUM_HIDDEN_NODES = 512
        LSTM_UNITS = 256
        conv3 = animal_conv_residual_fixup(inputs, depths = [16,32,64,128])
        flatten = tf.contrib.layers.flatten(inputs = conv3)

        hidden0 = tf.layers.dense(inputs=vels_ph, units=32, activation=tf.nn.elu)
        hidden1 = tf.layers.dense(inputs=flatten, units=NUM_HIDDEN_NODES, activation=tf.nn.elu)

        
        dones_ph = tf.placeholder(tf.bool, [batch_num])
        states_ph = tf.placeholder(tf.float32, [env_num, 2*LSTM_UNITS])
        hidden2 = tf.concat([hidden0, hidden1], axis=-1)
        lstm_out, lstm_state, initial_state = openai_lstm('lstm_ac', hidden2, dones_ph=dones_ph, states_ph=states_ph, units=LSTM_UNITS, env_num=env_num, batch_num=batch_num)
        value = tf.layers.dense(inputs=lstm_out, units=1, activation=None)
        if continuous:
            mu = tf.layers.dense(inputs=lstm_out, units=actions_num, activation=tf.nn.tanh)
            var = tf.layers.dense(inputs=lstm_out, units=actions_num, activation=tf.nn.softplus)
            return mu, var, value, states_ph, vels_ph, dones_ph, lstm_state, initial_state
        else:
            logits = tf.layers.dense(inputs=lstm_out, units=actions_num, activation=None)
            return logits, value, states_ph, vels_ph, dones_ph, lstm_state, initial_state