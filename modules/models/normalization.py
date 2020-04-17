
# File:           normalization.py
# COMMENTS:       This file includes batch normalization module.
# AUTHOR:         Rui Shu.
# PAPER:          not applicable.
# REPOSITORY:     https://github.com/RuiShu/tensorbayes


import tensorflow as tf
from tensorflow.contrib.framework import add_arg_scope

def _assign_moving_average(orig_val, new_val, momentum, name):
    with tf.name_scope(name):
        scaled_diff = (1 - momentum) * (new_val - orig_val)
        return tf.assign_add(orig_val, scaled_diff)

@add_arg_scope
def batch_norm(x, phase, shift=True, scale=True, momentum=0.99, eps=1e-7, scope=None, reuse=None):

    C = x._shape_as_list()[-1]
    ndim = len(x.shape)
    var_shape = [1] * (ndim - 1) + [C]

    with tf.variable_scope(scope, 'batch_norm', reuse=reuse):
        def training():
            m, v = tf.nn.moments(x, list(range(ndim - 1)), keep_dims=True)
            
            update_m = _assign_moving_average(moving_m, m, momentum, 'update_mean')
            update_v = _assign_moving_average(moving_v, v, momentum, 'update_var')
            tf.add_to_collection('update_ops', update_m)
            tf.add_to_collection('update_ops', update_v)

            with tf.control_dependencies([update_m, update_v]):
                output = (x - m) * tf.rsqrt(v + eps)

            return output

        def testing():
            m, v = moving_m, moving_v
            output = (x - m) * tf.rsqrt(v + eps)
            return output

        # Get mean and variance, normalize input
        moving_m = tf.get_variable('mean', var_shape, initializer=tf.zeros_initializer, trainable=False)
        moving_v = tf.get_variable('var', var_shape, initializer=tf.ones_initializer, trainable=False)

        if isinstance(phase, bool):
            output = training() if phase else testing()
        else:
            output = tf.cond(phase, training, testing)

        if scale:
            output *= tf.get_variable('gamma', var_shape, initializer=tf.ones_initializer)

        if shift:
            output += tf.get_variable('beta', var_shape, initializer=tf.zeros_initializer)

    return output