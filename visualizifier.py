import numpy as np
import umap.umap_ as umap
from matplotlib import pyplot as plt

umap_neighbors = 15
umap_dim = 3

channels = [
    'Thermistor','Respiratory Belt','ECG'
]

paths = [
    '_Supervised CNN_train_features_sub_1to50.npy', '_Supervised CNN_validation_features_sub_1to50.npy', '_Supervised CNN_test_features_sub_1to50.npy'
]

features = 'CNN'

if __name__ == '__main__':
    if umap_dim == 3:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=False, subplot_kw=dict(projection="3d"))
    elif umap_dim == 2:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=False)
    
    ax = [ax1, ax2, ax3]
   
    for i, channel in enumerate(channels):
        print('Loading ', channel)
        f = np.load(f'{channel}_{features}_features_sub_50to100.npy', allow_pickle=True)

        print('Feature shape: ', f.shape)

        reducer = umap.UMAP(n_neighbors=umap_neighbors, n_components=umap_dim)
        embedding = reducer.fit_transform(f)

        ax[i].set_title(channel)
        if umap_dim==2:
            ax[i].scatter(embedding[:,0], embedding[:,1], c='maroon', marker='.')
        else:
            ax[i].scatter(embedding[:,0], embedding[:,1], embedding[:,2], marker='.', c='maroon')

    plt.savefig('features_umap_sub50-100.png')

    if umap_dim == 3:
        fig, ax = plt.subplots(3, 3, sharey=False, subplot_kw=dict(projection="3d"))
    elif umap_dim == 2:
        fig, ax = plt.subplots(3, 3, sharey=False)

    for i, channel in enumerate(channels):
        for j, path in enumerate(paths):
            print(f'{i}:{j} {channel}')
            f = np.load(channel+path, allow_pickle=True)
            if 'train' in path:
                y = np.load('data/first 50/y_train.npy')
            elif 'validation' in path:
                y = np.load('data/first 50/y_valid.npy')
            elif 'test' in path:
                y = np.load('data/first 50/y_test.npy')
            else:
                print('I don\'t understand Python')

            print('Feature shape: ', f.shape)
            print('Label shape: ', y.shape)

            reducer = umap.UMAP(n_neighbors=umap_neighbors, n_components=umap_dim)
            embedding = reducer.fit_transform(f)

            ax[i][j].set_title(channel)
            if umap_dim==2:
                ax[i][j].scatter(embedding[:,0], embedding[:,1], c=y, marker='.')
            else:
                ax[i][j].scatter(embedding[:,0], embedding[:,1], embedding[:,2], marker='.', c=y)

    plt.savefig('features_umap_sub0-50.png')


