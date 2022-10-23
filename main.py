import numpy as np
from model_wrappers import NNCLR_C
from sklearn.model_selection import train_test_split
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

chan_dic = {
    'ECG' : [8],
    'Thermistor': [9],
    'Respiratory Belt' : [10, 11]
}


if __name__ == '__main__':
    print('Reading subjects 50-100...')
    x_train = np.load('data/second 50/x_train.npy')
    x_val = np.load('data/second 50/x_valid.npy')
    x_test = np.load('data/second 50/x_test.npy')

    x_all = np.concatenate((x_train, x_val, x_test), axis=0)
    
    print('X train shape: ', x_train.shape)
    print('X val shape: ', x_val.shape)
    print('X test shape: ', x_test.shape)

    #x_all = x_all[:,:,[ECG, FLOW, THO, ABD]]
    #Swapping to channels first
    x_all = np.moveaxis(x_all, 2, 1)
    print('X all shape: ', x_all.shape) #expected 90210

    

    x_all_train, x_all_val = train_test_split(x_all, test_size=0.2, random_state=1899, shuffle=True)
    for key in chan_dic.keys():
        print("Channel: ", key)
        isolated_channel_train = x_all_train[:,chan_dic[key],:]
        isolated_channel_val = x_all_val[:,chan_dic[key],:]
        print('Isolated channel train shape: ', isolated_channel_train.shape)
        print('Isolated channel validation shape: ', isolated_channel_val.shape)
        
        feature_learner = NNCLR_C(isolated_channel_train, np.ones((isolated_channel_train.shape[0])))
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

    #clean up "second 50" samples
    #load "first 50 samples"

    print('Reading subjects 50-100...')
    x_train = np.load('data/first 50/x_train.npy')
    x_val = np.load('data/first 50/x_valid.npy')
    x_test = np.load('data/second 50/x_test.npy')
    
    for key in chan_dic.keys():
        continue #skip until real files are loaded
        print("Channel: ", key)
        
        #load each per-channel feature learner
        feature_learner = NNCLR_C(X_train[:,chan_dic[key],:], np.ones((isolated_channel_train.shape[0])))
        feature_learner.model.load_state_dict(torch.load( f'{key}_feature_learner_weights.pt'))
        torch.load(f'{key}_feature_learner_weights.pth')
        #write a feature set for part of the first-50 set
        f = feature_learner.get_features(X_train[:,chan_dic[key],:])
        print("Train Feature shape: ", f.shape)
        np.save(f'{key}_train_features_sub_1to50.npy', f)

        f = feature_learner.get_features(X_val[:,chan_dic[key],:])
        print("Validation Feature shape: ", f.shape)
        np.save(f'{key}_validation_features_sub_1to50.npy', f)

        f = feature_learner.get_features(X_test[:,chan_dic[key],:])
        print("Test Feature shape: ", f.shape)
        np.save(f'{key}_test_features_sub_1to50.npy', f)

    print("Fin")
        


