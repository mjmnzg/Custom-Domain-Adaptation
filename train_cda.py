#   Paper: Custom Domain Adaptation: a new method for cross-subject, EEG-based cognitive load recognition
#   Authors: Magdiel Jiménez-Guarneros, Pilar Gómez-Gil
#   Contact emails: magdiel.jg@inaoep.mx, mjmnzg@gmail.com
#   National Institute Of Astrophysics, Optics and Electronics, Puebla, Mexico
#   Python-v3.6, Tensorflow-v.1.9

from modules.util import config_logging, collect_vars, batch_generator
from modules.models.model import get_model_fn
from modules.models.losses import mmd_loss
import logging
import os
import random
from collections import deque
import numpy as np
import tensorflow as tf
from tqdm import tqdm

def cda_trainer(Sx_train, Tx_train, Ty_train, Tx_test, Ty_test,
            model, 
            output,
            iterations = 50,
            batch_size = 64,
            display = 1,
            lr = 1e-4,
            weights = None,
            solver='sgd', 
            seed=1234, 
            n_classes = 4, 
            stepsize = 10, 
            weight_decay=0.1,
            weight_factor=0.1,
            gpu = '0'):
    """
                Module to train Deep neural network.

                Params
                    Sx_train: Training samples of the Source Domain.
                    Tx_train: Samples of the target domain used for domain adaptation.
                    Ty_train: Ground truth of the samples in target domain used for domain adaptation.
                    Tx_test: Samples of the target domain used for testing.
                    Ty_test: Ground truth of the samples in target domain used for testing.
                    model: name of the deep neural network model to be used.
                    output: output folder.
                    iterations: number of epochs.
                    batch_size: batch size.
                    display: how often the accuracy is displayed on the screen
                    lr: learning rate.
                    solver: training algorithm.
                    seed: random seed.
                    n_classes: number of classes.
                    stepsize: how often the weight decay is updated
                    weight_decay: weight decay.
                    weight_factor: weight factor used in domain adaptation.
                    gpu: number of gpu
            """
    # miscellaneous setup
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
    
    print("SEED:", seed)
    
    # display log in monitor
    logging.info('Adapting')
    
    # function with deep network model
    model_fn = get_model_fn(model)
    print("LOAD",model)
    # reset model
    tf.reset_default_graph()
    
    # input source data
    X_source = tf.placeholder(tf.float32, shape=(None, Sx_train.shape[1], Sx_train.shape[2], Sx_train.shape[3], Sx_train.shape[4]),name='source_input')
    # input target data
    X_target = tf.placeholder(tf.float32, shape=(None, Sx_train.shape[1], Sx_train.shape[2], Sx_train.shape[3], Sx_train.shape[4]),name='target_input')
    # input labels
    y = tf.placeholder(tf.int64, name="output")
    # input flag to enable training phase
    is_training = tf.placeholder(tf.bool, [])
    # probability for dropout
    prob = tf.placeholder_with_default(1.0, shape=())

    # create source model
    layers_source = model_fn(X_source, prob=prob, is_training=is_training, scope='source', num_outputs=n_classes)
    # create target model
    layers_target = model_fn(X_target, prob=prob, is_training=is_training, scope='target', num_outputs=n_classes)
    source_ft = layers_source["output"]
    target_ft = layers_target["output"]

    # Maximum Mean Discrepancy (MMD) loss to reduce the differences in the conditional distributions
    mapping_loss = weight_factor * mmd_loss(tf.nn.softmax(layers_source["output"]), tf.nn.softmax(layers_target["output"]))

    # variable collection
    source_vars_all = collect_vars('source')
    target_vars_all = collect_vars('target')

    # optimizer
    lr_var = tf.Variable(lr, name='learning_rate', trainable=False)
    
    # OPTIMIZER
    if solver == 'sgd':
        optimizer = tf.train.MomentumOptimizer(lr_var, 0.9, use_nesterov=True)
    elif solver == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(lr_var, 0.9)
    else:
        optimizer = tf.train.AdamOptimizer(lr_var)

    mapping_feat_step = optimizer.minimize(mapping_loss, var_list=list(target_vars_all.values()))


    # accuracy functions
    predictions_classifier_source = tf.argmax(source_ft, -1)
    accuracies_classifier_source = tf.reduce_mean(tf.cast(tf.equal(predictions_classifier_source, y), tf.float32))
    
    predictions_classifier_target = tf.argmax(target_ft, -1)
    accuracies_classifier_target = tf.reduce_mean(tf.cast(tf.equal(predictions_classifier_target, y), tf.float32))
    
    
    # set up session
    config = tf.ConfigProto(device_count=dict(GPU=1))
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    sess.run(tf.global_variables_initializer())
    
    # restore weights
    if os.path.isdir(weights):
        weights = tf.train.latest_checkpoint(weights)
    logging.info('Restoring weights from {}:'.format(weights))
    logging.info('    Restoring source model:')
    for src, tgt in source_vars_all.items():
        logging.info('        {:30} -> {:30}'.format(src, tgt.name))
    source_restorer = tf.train.Saver(var_list=source_vars_all)
    source_restorer.restore(sess, weights) # restores weights for SOURCE
    
    
    logging.info('    Restoring target model:')
    for src, tgt in target_vars_all.items():
        logging.info('        {:30} -> {:30}'.format(src, tgt.name))
    target_restorer = tf.train.Saver(var_list=target_vars_all)
    target_restorer.restore(sess, weights) # restores weights for TARGET

    # optimization
    output_dir = os.path.join(output,'')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    # save the losses
    mapping_losses = deque(maxlen=10)
    
    # bar progress
    bar = tqdm(range(iterations))
    bar.set_description('{} (lr: {:.0e})'.format(output, lr))
    bar.refresh()

    # calculate num of batches
    ntrain = Sx_train.shape[0]
    num_batch = ntrain//(batch_size)
    
    # Batch generators
    gen_source_batch = batch_generator([Sx_train], batch_size)
    gen_target_batch = batch_generator([Tx_train], batch_size)
    
    for i in bar:
        for _ in range(num_batch):
            
            X0 = gen_source_batch.__next__()
            X1 = gen_target_batch.__next__()
            
            # data and hyper-parameters
            feed = {X_source: X0[0], X_target: X1[0], prob:0.5, is_training:True}

            # update weights of deep neural network
            _ = sess.run(mapping_feat_step, feed_dict=feed)

            # get loss value
            mapping_loss_val = sess.run(mapping_loss, feed_dict=feed)

        # add loss value
        mapping_losses.append(mapping_loss_val)
        
        if i % display == 0:
            
            # Calculate accuracy on target training dataset using source network
            train_source_accuracy = sess.run(
                    accuracies_classifier_source,
                    feed_dict={X_source: Tx_train,
                               X_target: Tx_train,
                               y: Ty_train,
                               is_training:False,
                               prob:1.0})

            # Calculate accuracy on target testing dataset using source network
            test_source_accuracy = sess.run(
                accuracies_classifier_source,
                feed_dict={X_source: Tx_test,
                           X_target: Tx_test,
                           y: Ty_test,
                           is_training: False,
                           prob: 1.0})

            # Calculate accuracy on target training dataset using target network
            train_target_accuracy = sess.run(
                accuracies_classifier_target,
                feed_dict={X_source: Tx_train,
                           X_target: Tx_train,
                           y: Ty_train,
                           is_training: False,
                           prob: 1.0})

            # Calculate accuracy on target testing dataset using target network
            test_target_accuracy = sess.run(
                    accuracies_classifier_target,
                    feed_dict={X_source: Tx_test,
                               X_target: Tx_test,
                               y: Ty_test,
                               is_training:False,
                               prob:1.0})

            
            # Show in SCREEN
            logging.info('{:20} loss: {:10.4f}  acc_tr_sc:{:10.4f}  acc_ts_sc:{:10.4f} acc_tr_tg:{:10.4f}  acc_ts_tg:{:10.4f}'.format('Iteration {}:'.format(i+1),
                np.mean(mapping_losses),
                train_source_accuracy,
                test_source_accuracy,
                train_target_accuracy,
                test_target_accuracy
            ))

        # apply weight decay
        if stepsize is not None and (i + 1) % stepsize == 0:
            lr = sess.run(lr_var.assign(lr * weight_decay))
            logging.info('Changed learning rate to {:.0e}'.format(lr))
            bar.set_description('{} (lr: {:.0e})'.format(output, lr))

    # save deep neural network model
    snapshot_path = target_restorer.save(sess, output_dir, global_step=i+1)
    logging.info('Saved snapshot to {}'.format(snapshot_path))
    
    list_metrics_classification = []
    

    # Calculate final accuracies
    # Calculate accuracy on target training dataset using target network
    y_preds_train_target, train_target_accuracy = sess.run([predictions_classifier_target, accuracies_classifier_target],
            feed_dict={X_source: Tx_train,
                       X_target: Tx_train,
                       y: Ty_train,
                       is_training:False,
                       prob:1.0})

    # Calculate accuracy on target testing dataset using target network
    y_preds_test_target, test_target_accuracy = sess.run([predictions_classifier_target, accuracies_classifier_target],
            feed_dict={X_source: Tx_test,
                       X_target: Tx_test,
                       y: Ty_test,
                       is_training:False,
                       prob:1.0})
    print("accuracy_train_target:", train_target_accuracy, "    accuracy_test_target:", test_target_accuracy)

    # metrics to classification
    list_metrics_classification.append(train_target_accuracy)
    list_metrics_classification.append(test_target_accuracy)

    coord.request_stop()
    coord.join(threads)
    sess.close()
    
    return list_metrics_classification
