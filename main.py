import numpy as np
from model_wrappers import NNCLR_C
from sklearn.model_selection import ShuffleSplit
import torch

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

    x_all = x_all[:,:,[ECG, FLOW, THO, ABD]]
    x_all = np.moveaxis(x_all, 2, 1)
    print('X all shape: ', x_all.shape) #expected 90210

    rs = ShuffleSplit(n_splits=1, test_size=.2, random_state=1899)
    splits = rs.split(x_all)

    for train_index, test_index in splits:
        x_all_train = x_all[train_index]
        x_all_val = x_all[test_index]
    feature_learner = NNCLR_C(x_all_train, np.ones((x_all_train.shape[0])))
    feature_learner.fit(
        x_all_train, np.ones(x_all_train.shape[0]),x_all_val, np.ones(x_all_val.shape[0])
    )

    torch.save(feature_learner, 'feature_learner_weights.pth')


