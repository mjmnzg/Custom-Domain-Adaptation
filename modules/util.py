import logging
import logging.config
import os.path
from collections import OrderedDict
import numpy as np
import tensorflow as tf
import yaml
from tqdm import tqdm

class TqdmHandler(logging.StreamHandler):

    def __init__(self):
        logging.StreamHandler.__init__(self)

    def emit(self, record):
        msg = self.format(record)
        tqdm.write(msg)


def config_logging(logfile=None):
    path = os.path.join(os.path.dirname(__file__), 'logging.yml')
    with open(path, 'r') as f:
        config = yaml.load(f.read())
    if logfile is None:
        del config['handlers']['file_handler']
        del config['root']['handlers'][-1]
    else:
        config['handlers']['file_handler']['filename'] = logfile
    logging.config.dictConfig(config)


def remove_first_scope(name):
    return '/'.join(name.split('/')[1:])


def collect_vars(scope, start=None, end=None, prepend_scope=None):
    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
    var_dict = OrderedDict()
    if isinstance(start, str):
        for i, var in enumerate(vars):
            var_name = remove_first_scope(var.op.name)
            if var_name.startswith(start):
                start = i
                break
    if isinstance(end, str):
        for i, var in enumerate(vars):
            var_name = remove_first_scope(var.op.name)
            if var_name.startswith(end):
                end = i
                break
    for var in vars[start:end]:
        var_name = remove_first_scope(var.op.name)
        
        if prepend_scope is not None:
            var_name = os.path.join(prepend_scope, var_name)
        
        #names = var.op.name.split('/')[2:3]
        #if names[0] != "BatchNorm":
        var_dict[var_name] = var
    
    return var_dict


def shuffle_aligned_list(data):
    """Shuffle arrays in a list by shuffling each array identically."""
    num = data[0].shape[0]
    p = np.random.permutation(num)
    return [d[p] for d in data]


def batch_generator(data, batch_size, shuffle=True):
    """Generate batches of data.
    
    Given a list of array-like objects, generate batches of a given
    size by yielding a list of array-like objects corresponding to the
    same slice of each input.
    """
    if shuffle:
        data = shuffle_aligned_list(data)

    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size >= len(data[0]):
            batch_count = 0

            if shuffle:
                data = shuffle_aligned_list(data)

        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield [d[start:end] for d in data]
    
def countable_parameters():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        
        total_parameters += variable_parameters
    return total_parameters
