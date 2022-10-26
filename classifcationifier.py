import numpy as np
import umap.umap_ as umap
from matplotlib import pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

channels = [
    'Thermistor','Respiratory Belt','ECG'
]

if __name__ == '__main__':
    for channel in channels:
        x_train = np.load(f'{channel}_train_features_sub_1to50.npy')
        x_val = np.load(f'{channel}_validation_features_sub_1to50.npy')
        x_test = np.load(f'{channel}_test_features_sub_1to50.npy')
        y_train = np.load('data/first 50/y_train.npy')
        y_train = np.argmax(y_train, axis=-1)
        y_val = np.load('data/first 50/y_valid.npy')
        y_val = np.argmax(y_val, axis=-1)
        y_test = np.load('data/first 50/y_test.npy')
        y_test = np.argmax(y_test, axis=-1)

        #The test array labels are the wrong shape?
        if y_test.shape[0] < x_test.shape[0]:
            y_test = np.concatenate((y_test, np.zeros((x_test.shape[0] - y_test.shape[0]))))
        elif y_test.shape[0] > x_test.shape[0]:
            y_test = y_test[:x_test.shape[0]]

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

