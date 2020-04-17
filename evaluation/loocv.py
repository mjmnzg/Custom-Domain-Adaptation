# File:             loocv.py
# AUTHOR:           Pouya Bashivan
# PAPER:            Learning Representations from EEG with Deep Recurrent-Convolutional
#                   Neural Networks. International Conference on Learning Representations
#                   (ICLR) 2016.
# REPOSITORY:       https://github.com/pbashivan/EEGLearn

import numpy as np

def loocv(domains):
    """
    Generate Leave-Subject-Out cross validation
    """
    
    fold_pairs = []
    
    for i in np.unique(domains):
        #print(i)
        ts = domains == i       #return array with True where the index i is equal to indices in subjNumbers
        tr = np.squeeze(np.nonzero(np.bitwise_not(ts))) #first return array with Trues where the index i is equal to indices in subjNumbers but inverted
                                                        #after convert this array of numbers.
        ts = np.squeeze(np.nonzero(ts))                 #conver ts with trues to array with numbers
        
        
        np.random.shuffle(tr)       # Shuffle indices
        np.random.shuffle(ts)
        fold_pairs.append((tr, ts))
    
    
    return fold_pairs




