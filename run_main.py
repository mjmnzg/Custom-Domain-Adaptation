#!/usr/bin/python3

#   Paper: Custom Domain Adaptation: a new method for cross-subject, EEG-based cognitive load recognition
#   Authors: Magdiel Jiménez-Guarneros, Pilar Gómez-Gil
#   Contact emails: magdiel.jg@inaoep.mx, mjmnzg@gmail.com
#   National Institute Of Astrophysics, Optics and Electronics, Puebla, Mexico
#   Python-v3.6, Tensorflow-v.1.9

#   Commands to execute this code:
#   Pre-training
#    CUDA_VISIBLE_DEVICES=0 python3 run_main.py --model recresnet --dir_output model/recresnet --dir_resume outputs/resume --seed 223
#   Training Custom Domain Adaptation
#    CUDA_VISIBLE_DEVICES=0 python3 run_main.py --model cda --dir_output model/cda --dir_pretrain model/recresnet --dir_resume outputs/resume --seed 223

    
import numpy as np
from preprocessing.modules_pbashivan import load_bashivan_data
from evaluation.loocv import loocv
from preprocessing.modules_pbashivan import split_cognitive_load_data
from train_cda import cda_trainer
from train_dnn import dnn_trainer
import argparse
import random
import tensorflow as tf
import os

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="cnnmix", help='name of model to execute')
parser.add_argument('--dir_output', type=str, default="outputs/recresnet", help='name of output folder results')
parser.add_argument('--dir_pretrain', type=str, default="outputs/recresnet", help='folder used by DNN')
parser.add_argument('--dir_resume', type=str, default="outputs/resume", help='folder for resume')
parser.add_argument('--seed', type=int, default=223, help='seed')
args = parser.parse_args()


def losocv_cognitive_load(X, Y, subjects, args):
    """
        Leave One-Subject-Out Cross-Validation (LOOCV) on Cognitive Load data

        Params
            X: dataset containing all subject samples
            Y: dataset containing all subject labels
            subjects: dataset containing pairs between sample indexes and subjects
            args: hyper-parameters to train Custom Domain Adaptation.
    """

    # variable used to save accuracy results
    list_metrics_clsf = []
        
    # Extract pairs between indexes and subjects
    fold_pairs = loocv(subjects)
    
    # Iterate over fold_pairs
    for foldNum, fold in enumerate(fold_pairs):
        print('Beginning fold {0} out of {1}'.format(foldNum+1, len(fold_pairs)))
        
        # Divide dataset into training, validation and testing sets
        (Sx_train, Sy_train), (Sx_valid, Sy_valid), (Tx_train, Ty_train), (Tx_test, Ty_test), d_train, y_classes = split_cognitive_load_data(X, Y, subjects, fold, args.seed)

        # data shape
        print("Sx_train-shape:", Sx_train.shape, "Sx_valid-shape:", Sx_valid.shape)
        print("Tx_train-shape:", Tx_train.shape, "Tx_test-shape:", Tx_test.shape)
        print("y_classes:", y_classes)

        if args.model == "recresnet":

            # Hyper-parameters
            # ----------------
            # iterations = 50
            # batch size = 64
            # learning rate = 0.0001
            # solver = 'adam'
            # stepsize = 10
            # weight decay = 0.99

            classification_metrics = dnn_trainer(Sx_train, Sy_train, Sx_valid, Sy_valid, Tx_train, Ty_train, Tx_test, Ty_test,
                   model="recresnet",
                   output=args.dir_output + "/sub_" + str(foldNum + 1),
                   iterations=50,
                   seed=args.seed,
                   batch_size=64,
                   display=1,
                   lr=0.0001,
                   weight_decay=0.99,
                   solver="adam",
                   n_classes=len(y_classes),
                   stepsize=10)

        elif args.model == "cda":
            
            # Hyper-parameters
            # ----------------
            # Pre-trained RecResNet is required
            # iterations = 50
            # batch size = 64
            # learning rate = 0.001
            # solver = 'sgd' with momentum
            # weight decay = 0.99
            # stepsize = 10
            # weight factor = 0.1
            
            classification_metrics = cda_trainer(Sx_train, Tx_train, Ty_train, Tx_test, Ty_test,
                model="recresnet",
                output=args.dir_output+"/sub_"+str(foldNum+1),
                seed=args.seed,
                iterations=30,
                batch_size=64,
                display=1, 
                lr=0.001,
                weights=args.dir_pretrain+"/sub_"+str(foldNum+1), 
                solver="sgd",
                n_classes=len(y_classes),
                weight_decay=0.99,
                weight_factor=0.1)

        else:
            raise Exception("Unknown model %s." % args.model)
        
        # add to list
        list_metrics_clsf.append(classification_metrics)
        print()
    
    # To np array
    list_metrics_clsf = np.array(list_metrics_clsf)

    # determine mean accuracy
    print("CLASSIFICATION METRICS:")
    for i in range(len(list_metrics_clsf[0])):
        mean = list_metrics_clsf[:,i].mean()
        print("Metric [", (i+1), "] = ", list_metrics_clsf[:, i], " Mean:", mean)

    # Save Classification Metrics
    save_file = args.dir_resume+"/custom-domain-adaptation-classification-results.csv"
    f=open(save_file, 'ab')
    np.savetxt(f, list_metrics_clsf, delimiter=",", fmt='%0.4f')
    f.close()
    


def main(args):
    # set random seeds
    np.random.seed(args.seed)
    random.seed(args.seed)
    tf.set_random_seed(args.seed)
    print("SEED:", args.seed)
    
    # Create output folder
    if not os.path.isdir(args.dir_output):
        os.makedirs(args.dir_output)

    # Load public cognitive load dataset (Bashivan et al., 2016)
    # We used a window size of 24x24 (size_image) in this implementation, but a window size of 32x32 can be used as in
    # (Bashivan et al., 2016; Jiménez-Guarneros & Gómez-Gil, 2017).
    # The 'generate_images' options must be set to 'True' the first time to generate data samples.
    X, y, subjects = load_bashivan_data("/home/magdiel/Descargas/datasets/CognitiveLoad/",
                        n_channels=64, n_windows=7, n_bands=3, generate_images=False,
                        size_image=24, visualize=False)

    # run Leave One-Subject-Out Cross-Validation (LOSOCV).
    losocv_cognitive_load(X, y, subjects, args)


# Call main module
main(args)
