#   Paper: Custom Domain Adaptation: a new method for cross-subject, EEG-based cognitive load recognition
#   Authors: Magdiel Jiménez-Guarneros, Pilar Gómez-Gil
#   Contact emails: magdiel.jg@inaoep.mx, mjmnzg@gmail.com
#   National Institute Of Astrophysics, Optics and Electronics, Puebla, Mexico
#   Python-v3.6, Tensorflow-v.1.9

import tensorflow as tf
from collections import OrderedDict
from modules.models.model import register_model_fn
from modules.models.normalization import batch_norm

@register_model_fn('recresnet')
def recresnet(inputs, prob=0.5, is_training=True, scope='dnn', num_outputs=10, reuse=False):
    """
        Version of Residual Recurrent Network (RecResNet) in Tensorflow-v1.9.
        
        Jiménez-Guarneros M., Gómez-Gil P. "Cross-subject classification of cognitive loads using a recurrent-residual
        deep network". IEEE Symposium Series on Computational Intelligence (IEEE SSCI 2017).
        
        Original version was implemented in Theano library, but it is not commonly used now.

        Parameters:
            inputs - placeholder input data.
            prob - probability for dropout.
            scope - identifier to register deep neural network in execution.
            is_training - flag to enable training phase.
            reuse - flag to reuse network.
            num_outputs - number of output neurons.
    """
    layers = OrderedDict()
    
    with tf.variable_scope(scope, reuse=reuse):

        # CONV1
        conv1 = tf.layers.conv3d(inputs, filters=16, kernel_size=(1, 3, 3), strides=(1, 1, 1), padding='valid', activation=tf.nn.tanh)
        # AdaBN
        conv1 = batch_norm(conv1, phase=is_training)

        # CONV2
        conv2 = tf.layers.conv3d(conv1, filters=16, kernel_size=(1, 3, 3), strides=(1, 2, 2), padding='valid', activation=tf.nn.tanh)
        # AdaBN
        conv2 = batch_norm(conv2, phase=is_training)

        # Residual block
        step1 = tf.layers.conv3d(conv2, filters=16, kernel_size=(1, 3, 3), strides=(1, 1, 1), padding='same', activation=None)
        step1 = batch_norm(step1, phase=is_training)

        step2 = tf.nn.relu(step1)
        step3 = tf.layers.conv3d(step2, filters=16, kernel_size=(1, 3, 3), strides=(1, 1, 1), padding='same', activation=None)
        step4 = batch_norm(step3, phase=is_training)

        residual = tf.nn.relu(conv2 + step4)

        # Reshape layer
        nsamples = int(residual.get_shape()[1])
        nfeatures = int(residual.get_shape()[2]*residual.get_shape()[3]*residual.get_shape()[4])
        residual = tf.reshape(residual, [-1, nsamples, nfeatures])

        # default GRU
        cell = tf.contrib.rnn.GRUCell(128)
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=prob)
        cell = tf.contrib.rnn.MultiRNNCell([cell])
        rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell, residual, dtype=tf.float32)
        
        # Flatten
        rnn_out = tf.contrib.layers.flatten(rnn_outputs)

        # FC1
        fc1 = tf.contrib.layers.fully_connected(rnn_out, 256, activation_fn=tf.nn.relu)
        # AdaBN
        fc1 = batch_norm(fc1, phase=is_training)

        # dropout
        fc1 = tf.layers.dropout(fc1, rate=prob, training=is_training)
        # FC2
        fc2 = tf.contrib.layers.fully_connected(fc1, 256, activation_fn=tf.nn.relu)
        # AdaBN
        fc2 = batch_norm(fc2, phase=is_training)

        # Dropout
        fc2 = tf.layers.dropout(fc2, rate=prob, training=is_training)
        # Output
        layers["output"] = tf.contrib.layers.fully_connected(fc2, num_outputs, activation_fn=None)

    return layers