import numpy as np
import umap.umap_ as umap
from matplotlib import pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

# channels = [
#     'Thermistor','Respiratory Belt','ECG'
# ]

channels = [
    'X',
    'Y',
    'Z',
    'X, Y, Z together',
    'RMS'
]

if __name__ == '__main__':
    for channel in channels:
        x_train = np.load(f'{channel}_Twristar_CNN_train_features_sub_1to50.npy')
        x_val = np.load(f'{channel}_Twristar_CNN_validation_features_sub_1to50.npy')
        x_test = np.load(f'{channel}_Twristar_CNN_test_features_sub_1to50.npy')

        #y_train = np.load('data/first 50/y_train.npy')
        y_train = np.load('data/twristar/labeled/y_train.npy')

        #y_val = np.load('data/first 50/y_valid.npy')
        y_train, y_test = train_test_split(y_train, test_size=0.1)
        #y_test = np.load('data/first 50/y_test.npy')
        y_train, y_val = train_test_split(y_train, test_size=0.1)

        print('X train shape: ', x_train.shape)
        print('X val shape: ', x_val.shape)
        print('X test shape: ', x_test.shape)

        print('y train shape: ', y_train.shape)
        print('y val shape: ', y_val.shape)
        print('y test shape: ', y_test.shape)

        neigh = KNeighborsClassifier(n_neighbors=3)
        neigh.fit(x_train, y_train)
        y_pred = neigh.predict(x_test)

        print(confusion_matrix(y_test, y_pred))
        print(f"{channel} Accuracy: ", accuracy_score(y_test, y_pred))

