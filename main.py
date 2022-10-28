import numpy as np
from model_wrappers import NNCLR_R
from models.backbones import FCN
import torch
from torch import nn
import os
import gc

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

chan_dic = {
    'Thermistor': [9],
    'Respiratory Belt' : [10, 11],
    'ECG' : [8]
}


if __name__ == '__main__':
    if not os.path.exists('temp'):
        os.mkdir('temp')
    if not os.path.exists('results'):
        os.mkdir('results')
    print('Reading subjects 50-100...')
    x_train_second = np.load('data/second 50/x_train.npy', allow_pickle=True)
    x_val_second = np.load('data/second 50/x_valid.npy', allow_pickle=True)
    x_test_second = np.load('data/second 50/x_test.npy', allow_pickle=True)

    x_all = np.concatenate((x_train_second, x_val_second), axis=0)

    print('Second X train shape: ', x_train_second.shape)
    print('Second X val shape: ', x_val_second.shape)
    print('Second X test shape: ', x_test_second.shape)
    del x_train_second
    del x_val_second
    gc.collect()

    #x_all = x_all[:,:,[ECG, FLOW, THO, ABD]]
    #Swapping to channels first
    x_all = np.moveaxis(x_all, 2, 1)
    x_test_second = np.moveaxis(x_test_second, 2, 1)
    print('X all shape: ', x_all.shape) #expected 90210

    print('Reading subjects 0-49...')
    x_train_first = np.load('data/first 50/x_train.npy', allow_pickle=True)
    x_val_first = np.load('data/first 50/x_valid.npy', allow_pickle=True)
    x_test_first = np.load('data/first 50/x_test.npy', allow_pickle=True)

    X_train_first = np.moveaxis(x_train_first, 2, 1)
    X_val_first = np.moveaxis(x_val_first, 2, 1)
    X_test_first = np.moveaxis(x_test_first, 2, 1)

    for key in chan_dic.keys():
        print("Channel: ", key)
        isolated_channel_train = x_all[:,chan_dic[key],:]
        isolated_channel_val = x_test_second[:,chan_dic[key],:]
        print('Isolated channel train shape: ', isolated_channel_train.shape)
        print('Isolated channel validation shape: ', isolated_channel_val.shape)
        
        feature_learner = NNCLR_R(isolated_channel_train, np.ones((isolated_channel_train.shape[0])))
        feature_learner.fit(
            isolated_channel_train, np.ones(isolated_channel_train.shape[0]),
            isolated_channel_val, np.ones(isolated_channel_val.shape[0])
        )
        torch.save(feature_learner.model.state_dict(), f'{key}_feature_learner_weights.pt')
        
        f0 = feature_learner.get_features(isolated_channel_train)
        f1 = feature_learner.get_features(isolated_channel_val)
        f = np.concatenate((f0, f1), axis=0)
        print("Feature shape: ", f.shape)
        np.save(f'{key}_features_sub_50to100.npy', f)

        #write a feature set for part of the first-50 set
        f = feature_learner.get_features(x_train_first[:,chan_dic[key],:])
        print("Train Feature shape: ", f.shape)
        np.save(f'{key}_train_features_sub_1to50.npy', f)

        f = feature_learner.get_features(x_val_first[:,chan_dic[key],:])
        print("Validation Feature shape: ", f.shape)
        np.save(f'{key}_validation_features_sub_1to50.npy', f)

        f = feature_learner.get_features(x_test_first[:,chan_dic[key],:])
        print("Test Feature shape: ", f.shape)
        np.save(f'{key}_test_features_sub_1to50.npy', f)

    print("Fin")
        


