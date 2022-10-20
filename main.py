import numpy as np
from model_wrappers import NNCLR_R

EEG_A1_A2 = 0
EEG_C3_A2 = 1
EEG_C4_A1 = 2
EOG_LEFT = 3
EOG_RIGHT = 4
EMG_CHIN = 5
LEG_1 = 6
LEG_2 = 7
ECG = 8
FLOW = 9
THO = 10
ABD = 11


if __name__ == '__main__':
    print('Reading arrays...')
    x_train = np.load('data/x_train.npy')
    x_val = np.load('data/x_valid.npy')
    x_test = np.load('data/x_test.npy')

    x_all = np.concatenate((x_train, x_val, x_test), axis=0)
    
    print('X train shape: ', x_train.shape)
    print('X val shape: ', x_val.shape)
    print('X test shape: ', x_test.shape)
    #print('X all shape: ', x_all.shape) #expected 90210

    x_all = x_all[:,:,[ECG, FLOW, THO, ABD]]
    #print('X all shape: ', x_all.shape) #expected 90210
    x_all = np.moveaxis(x_all, 2, 1)
    print('X all shape: ', x_all.shape) #expected 90210

    feature_learner = NNCLR_R(x_all, np.zeros((x_all.shape[0])))


