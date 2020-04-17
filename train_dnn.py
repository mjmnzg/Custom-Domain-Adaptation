#   Paper: Custom Domain Adaptation: a new method for cross-subject, EEG-based cognitive load recognition
#   Authors: Magdiel Jiménez-Guarneros, Pilar Gómez-Gil
#   Contact emails: magdiel.jg@inaoep.mx, mjmnzg@gmail.com
#   National Institute Of Astrophysics, Optics and Electronics, Puebla, Mexico
#   Python-v3.6, Tensorflow-v.1.9

from modules.util import config_logging, collect_vars, batch_generator
from modules.models.recresnet import recresnet
from modules.models.model import get_model_fn
import logging
import os
import random
from collections import deque
import numpy as np
import tensorflow as tf
from tqdm import tqdm


def dnn_trainer(Sx_train, Sy_train, Sx_valid, Sy_valid, Tx_train, Ty_train, Tx_test, Ty_test,
                model, 
                output,
                iterations = 50, 
                batch_size = 20, 
                display = 1, 
                lr = 1e-4,
                solver='sgd', 
                seed=1234, 
                n_classes = 4, 
                stepsize = 10,
                weight_decay=0.1,
                gpu = '0'):
    """
            Module to train Deep neural network.

            Params
                Sx_train: Training samples of the Source Domain.
                Sy_train: Training labels of the Source Domain.
                Sx_valid: Validation samples of the Source Domain.
                Sy_valid: Validation labels of the Source Domain.
                Tx_train: Samples of the target domain used for domain adaptation.
                Ty_train: Ground truth of the samples in target domain used for domain adaptation.
                Tx_test: Samples of the target domain used for testing.
                Ty_test: Ground truth of the samples in target domain used for testing.
                model: name of the deep neural network model to be used.
                output: output folder.
                iterations: number of epochs.
                batch_size: batch size.
                display: how often the accuracy is displayed on the screen
                lr = learning rate.
                solver: training algorithm.
                seed: random seed.
                n_classes: number of classes.
                stepsize: how often the weight decay is updated
                weight_decay: weight decay.
                gpu: number of gpu
        """

    config_logging()
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        logging.info('CUDA_VISIBLE_DEVICES specified, ignoring --gpu flag')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    logging.info('Using GPU {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))
    
    if seed is None:
        seed = random.randrange(2 ** 32 - 2)
    logging.info('Using random seed {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed + 1)
    tf.set_random_seed(seed + 2)
    
    # create path directory
    if not os.path.exists(output):
        os.mkdir(output)
        
    # obtain model 
    model_fn = get_model_fn(model)
    
    # reset model
    tf.reset_default_graph()
    
    # Create input variables
    # input data samples
    X = tf.placeholder(tf.float32, shape=(None, Sx_train.shape[1], Sx_train.shape[2], Sx_train.shape[3], Sx_train.shape[4]),name='Input')
    # input labels
    y = tf.placeholder(tf.int64)
    # flag to indicate training phase
    is_training = tf.placeholder(tf.bool, []) # flag training phase
    # probability for dropout
    prob = tf.placeholder_with_default(1.0, shape=()) # dropout
    
    # create model: im_batch is shape of images
    layers = model_fn(X, num_outputs=n_classes, scope=model, is_training=is_training, prob=prob)
    classifier_net = layers["output"]
    
    # L1 reguralizer (Optional)
    l1 = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    weight_factor = 0.01  # Choose an appropriate one.
    
    # Classification loss
    classifier_loss = tf.losses.sparse_softmax_cross_entropy(y, classifier_net) + weight_factor * sum(l1)

    # create learning rate variable
    lr_var = tf.Variable(lr, name='learning_rate', trainable=False)
    
    # Optimizer
    if solver == 'sgd':
        optimizer = tf.train.MomentumOptimizer(lr_var, 0.9, use_nesterov=True)
    elif solver == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(lr_var, 0.9)
    else:
        optimizer = tf.train.AdamOptimizer(lr_var)
    
    
    # step to update deep neural network
    step_classifier = optimizer.minimize(classifier_loss)
    
    # accuracy function
    predictions = tf.argmax(classifier_net, -1)
    accuracies = tf.reduce_mean(tf.cast(tf.equal(predictions, y), tf.float32))
    
    
    config = tf.ConfigProto(device_count=dict(GPU=1))
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    sess.run(tf.global_variables_initializer())
    
    # collect model variables
    model_vars = collect_vars(model)
    
    # save variables
    saver = tf.train.Saver(var_list=model_vars)
    output_dir = os.path.join(output, 'snapshot')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    # average loss
    avg_loss = deque(maxlen=10)
    
    # progress bar
    bar = tqdm(range(iterations))
    bar.set_description('{} (lr: {:.0e})'.format(output, lr))
    bar.refresh()

    # calculate number of batches
    ntrain = Sx_train.shape[0]
    num_batch = ntrain//(batch_size)

    # initial learning rate
    sess.run(lr_var.assign(lr))
    
    # Batch generators
    gen_source_batch = batch_generator([Sx_train, Sy_train], batch_size)
    
    # iterate on variable var
    for i in bar:
        for _ in range(num_batch):
            
            X0, y0 = gen_source_batch.__next__()
            
            # data
            feed = {X: X0, y: y0, is_training: True, prob: 0.5}
            # update weights of deep neural network model
            _, clsf_loss = sess.run([step_classifier, classifier_loss], feed_dict=feed)
            
        # append loss value
        avg_loss.append(clsf_loss)
            
        
        # display advance each 'display' iterations
        if i % display == 0:
            
            # Calculate accuracy on source validation dataset using deep neural network
            valid_source_accuracy = sess.run(accuracies, 
                    feed_dict={ X: Sx_valid, 
                                y: Sy_valid, 
                                is_training: False,
                                prob: 1.0})
            
            # Calculate accuracy on target training dataset using deep neural network
            train_target_accuracy = sess.run(accuracies, 
                    feed_dict={X: Tx_train, 
                               y: Ty_train, 
                               is_training: False,
                               prob:1.0})
                               
            # Calculate accuracy on target testing dataset using deep neural network
            test_target_accuracy = sess.run(accuracies,
                    feed_dict={X: Tx_test,
                               y: Ty_test,
                               is_training: False,
                               prob:1.0})
            
            # show accuracy in screen
            logging.info('{:20} loss: {:10.4f}  acc_val_sc:{:10.4f} acc_tr_tg:{:10.4f} acc_ts_tg:{:10.4f}'.format('Iteration {}:'.format(i+1),
                    np.mean(avg_loss),
                    valid_source_accuracy,
                    train_target_accuracy,
                    test_target_accuracy
                    ))
        
        # apply weight decay
        if stepsize is not None and (i + 1) % stepsize == 0:
            lr = sess.run(lr_var.assign(lr * weight_decay))
            logging.info('Changed learning rate to {:.0e}'.format(lr))
            bar.set_description('{} (lr: {:.0e})'.format(output, lr))
            
            
    # save deep neural network model
    snapshot_path = saver.save(sess, output_dir, global_step=i+1)
    logging.info('Saved snapshot to {}'.format(snapshot_path))
    
    list_metrics_classification = []
    

    # Get final accuracy
    acc_train_target, y_preds_train = sess.run([accuracies, predictions], feed_dict={X: Tx_train, y: Ty_train, is_training: False, prob: 1.0})
    acc_test_target, y_preds_test = sess.run([accuracies, predictions], feed_dict={X: Tx_test, y: Ty_test, is_training: False, prob: 1.0})
    print("accuracy_train_target:", acc_train_target, "    accuracy_test_target:", acc_test_target)

    # metrics to classification
    list_metrics_classification.append(acc_train_target)
    list_metrics_classification.append(acc_test_target)

    coord.request_stop()
    coord.join(threads)
    sess.close()
    
    return list_metrics_classification