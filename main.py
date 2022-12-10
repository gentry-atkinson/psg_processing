import numpy as np
from model_wrappers import NNCLR_C
from sklearn.model_selection import train_test_split
import torch
import os
import gc

# EEG_A1_A2 = 0
# EEG_C3_A2 = 1
# EEG_C4_A1 = 2
# EOG_LEFT = 3
# EOG_RIGHT = 4
# EMG_CHIN = 5
# LEG_1 = 6
# LEG_2 = 7
# ECG = 8
# FLOW = 9
# THO = 10
# ABD = 11

chan_dic = {
    'Thermistor Flow': [9],
    'Thorasic': [10],
    'Abdominal': [11],
    'ECG' : [8],
    'EOG' : [3, 4],
    'EMG' : [5],
    'Leg' : [6, 7]
} 

# chan_dic = {
#     'Accel' : [0],
#     'BVP' : [1],
#     'EDA' : [2],
#     'P Temp' : [3],
#     'All together' : [0, 1, 2, 3]
# }

# chan_dic = {
#     'test' : [0]
# }

features = 'psg_CNN'

NUM_CLASS = 2

UNLABELED_DIR = 'second 50'
LABELED_DIR = 'first 50'
# UNLABELED_DIR = 'twristar/unlabeled'
# LABELED_DIR = 'twristar/labeled'


if __name__ == '__main__':
    if not os.path.exists('temp'):
        os.mkdir('temp')
    if not os.path.exists('model_snapshots'):
        os.mkdir('model_snapshots')

    print('Reading (unlabeled) subjects 50-100...')
    x_train_second = np.load(f'data/{UNLABELED_DIR}/x_train.npy', allow_pickle=True)
    x_test_second = np.load(f'data/{UNLABELED_DIR}/x_test.npy', allow_pickle=True)
    x_val_second = np.load(f'data/{UNLABELED_DIR}/x_valid.npy', allow_pickle=True)
    #x_train_second = np.concatenate((np.zeros((500, 150, 1)), np.ones((300, 150, 1))), axis=0)
    #x_test_second = np.concatenate((np.zeros((500, 150, 1)), np.ones((300, 150, 1))), axis=0)
    #x_val_second = np.concatenate((np.zeros((500, 150, 1)), np.ones((300, 150, 1))), axis=0)
    

    # y_train_second = np.load('data/second 50/y_train.npy', allow_pickle=True)
    # y_val_second = np.load('data/second 50/y_valid.npy', allow_pickle=True)
    # y_test_second = np.load('data/second 50/y_test.npy', allow_pickle=True)

    # y_train_second = np.argmax(y_train_second, axis=-1)
    # y_val_second =  np.argmax(y_val_second, axis=-1)
    # y_test_second =  np.argmax(y_test_second, axis=-1)

    x_all_train = np.concatenate((x_train_second, x_val_second), axis=0)
    # y_all = np.concatenate((y_train_second, y_val_second), axis=0)

    print('Second X train shape: ', x_train_second.shape)
    print('Second X val shape: ', x_val_second.shape)
    print('Second X test shape: ', x_test_second.shape)
    del x_train_second
    del x_val_second
    gc.collect()

    #Swapping to channels first
    x_all_train = np.moveaxis(x_all_train, 2, 1)

    x_test_second = np.moveaxis(x_test_second, 2, 1)
    # x_val_second = np.moveaxis(x_val_second, 2, 1)
    # x_train_second = np.moveaxis(x_train_second, 2, 1)
    print('X all shape: ', x_all_train.shape)
    print('Second X test shape after move: ', x_test_second.shape)
    # print('Second X val shape after move: ', x_val_second.shape)
    # print('Second X train shape after move: ', x_train_second.shape)

    print('Reading (labeled) subjects 0-49...')
    x_train_first = np.load(f'data/{LABELED_DIR}/x_train.npy', allow_pickle=True)
    x_test_first = np.load(f'data/{LABELED_DIR}/x_test.npy', allow_pickle=True)
    x_val_first = np.load(f'data/{LABELED_DIR}/x_valid.npy', allow_pickle=True)
    #x_train_first = np.concatenate((np.zeros((500, 150, 1)), np.ones((300, 150, 1))), axis=0)
    #x_test_first = np.concatenate((np.zeros((500, 150, 1)), np.ones((300, 150, 1))), axis=0)
    #x_val_first = np.concatenate((np.zeros((500, 150, 1)), np.ones((300, 150, 1))), axis=0)


    x_train_first = np.moveaxis(x_train_first, 2, 1)
    x_val_first = np.moveaxis(x_val_first, 2, 1)
    x_test_first = np.moveaxis(x_test_first, 2, 1)

    x_all_train = np.concatenate((x_all_train, x_train_first), axis=0)

    print('First X train shape: ', x_train_first.shape)
    print('First X val shape: ', x_val_first.shape)
    print('First X test shape: ', x_test_first.shape)

    for key in chan_dic.keys():
        print("Channel: ", key)
        isolated_channel_train = x_all_train[:,chan_dic[key],:]
        isolated_channel_val = x_test_second[:,chan_dic[key],:]
        
        print('Isolated channel train shape: ', isolated_channel_train.shape)
        print('Isolated channel validation shape: ', isolated_channel_val.shape)
        
        
        #feature_learner = NNCLR_C(X=isolated_channel_train, y=y_train_second)
        #The y values are used to determine the number of classes
        feature_learner = NNCLR_C(X=isolated_channel_train, y=[1])
        feature_learner.fit(
            isolated_channel_train, np.ones(isolated_channel_train.shape[0]),
            isolated_channel_val, np.ones(isolated_channel_val.shape[0])
        )
        # feature_learner.fit(
        #     isolated_channel_train, y_all,
        #     isolated_channel_val, y_test_second
        # )

        torch.save(feature_learner.model.state_dict(), f'{key}_{features}_feature_learner_weights.pt')
        
        f0 = feature_learner.get_features(isolated_channel_train)
        f1 = feature_learner.get_features(isolated_channel_val)
        f = np.concatenate((f0, f1), axis=0)
        print("Feature shape: ", f.shape)
        np.save(f'{key}_{features}_features_sub_50to100.npy', f)

        #write a feature set for part of the first-50 set
        f = feature_learner.get_features(x_train_first[:,chan_dic[key],:])
        print("Train Feature shape: ", f.shape)
        np.save(f'{key}_{features}_train_features_sub_1to50.npy', f)

        f = feature_learner.get_features(x_val_first[:,chan_dic[key],:])
        print("Validation Feature shape: ", f.shape)
        np.save(f'{key}_{features}_validation_features_sub_1to50.npy', f)

        f = feature_learner.get_features(x_test_first[:,chan_dic[key],:])
        print("Test Feature shape: ", f.shape)
        np.save(f'{key}_{features}_test_features_sub_1to50.npy', f)

    print("Fin")
        


