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
    'Accel','BVP','EDA', 'P Temp', 'All together'
]

LABELED_DIR = 'twristar/labeled'

features = 'twristar_train-extract_on_same_50_CNN_'

K=3

if __name__ == '__main__':
    for channel in channels:
        x_train = np.load(f'{features}{channel}_train_features_sub_1to50.npy')
        x_val = np.load(f'{features}{channel}_validation_features_sub_1to50.npy')
        x_test = np.load(f'{features}{channel}_test_features_sub_1to50.npy')

        #y_train = np.load('data/first 50/y_train.npy')
        y_train = np.load(f'data/{LABELED_DIR}/y_train.npy')
        y_val = np.load(f'data/{LABELED_DIR}/y_valid.npy')
        y_test = np.load(f'data/{LABELED_DIR}/y_test.npy')

        if y_train.ndim>1:
            y_train = np.argmax(y_train, axis=-1)
        if y_val.ndim>1:
            y_val = np.argmax(y_val, axis=-1)
        if y_test.ndim>1:
            y_test = np.argmax(y_test, axis=-1)

        print('X train shape: ', x_train.shape)
        print('X val shape: ', x_val.shape)
        print('X test shape: ', x_test.shape)

        print('y train shape: ', y_train.shape)
        print('y val shape: ', y_val.shape)
        print('y test shape: ', y_test.shape)

        neigh = KNeighborsClassifier(n_neighbors=K)
        neigh.fit(x_train, y_train)
        y_pred = neigh.predict(x_test)

        print(confusion_matrix(y_test, y_pred))
        print(f"{channel} Accuracy: ", accuracy_score(y_test, y_pred))

