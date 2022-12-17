import numpy as np
import umap.umap_ as umap
from matplotlib import pyplot as plt

umap_neighbors = 15
umap_dim = 2



paths = [
    '_train_features_sub_1to50.npy', '_validation_features_sub_1to50.npy', '_test_features_sub_1to50.npy'
]

# chan_dic = {
#     'Accel' : [0],
#     'BVP' : [1],
#     'EDA' : [2],
#     'P Temp' : [3],
#     'All together' : [0, 1, 2, 3]
# }

chan_dic = {
    'Thermistor Flow': [9],
    'Thorasic': [10],
    'Abdominal': [11],
    'ECG' : [8],
    'EOG' : [3, 4],
    'EMG' : [5],
    'Leg' : [6, 7]
} 

features = 'psg_train_all_extract_labeled_CNN'

color = ['maroon', 'darkblue', 'gold', 'green']

# UNLABELED_DIR = 'twristar/unlabeled'
# LABELED_DIR = 'twristar/labeled'
UNLABELED_DIR = 'second 50'
LABELED_DIR = 'first 50'

MARKER_SIZE = 1

if __name__ == '__main__':
    # if umap_dim == 3:
    #     fig, ax = plt.subplots(1, len(chan_dic.keys()), sharey=False, subplot_kw=dict(projection="3d"), figsize=(3*len(chan_dic.keys()), 3))
    # elif umap_dim == 2:
    #     fig, ax = plt.subplots(1, len(chan_dic.keys()), sharey=False, figsize=(3*len(chan_dic.keys()), 3))
    
    # #ax = [ax1, ax2, ax3, ax4, ax5]
   
    # for i, channel in enumerate(chan_dic.keys()):
    #     print('Loading ', channel)
    #     f = np.load(f'{features}_{channel}_features_sub_50to100.npy', allow_pickle=True)

    #     print('Feature shape: ', f.shape)

    #     reducer = umap.UMAP(n_neighbors=umap_neighbors, n_components=umap_dim)
    #     embedding = reducer.fit_transform(f)

    #     ax[i].set_title(channel)
    #     if umap_dim==2:
    #         ax[i].scatter(embedding[:,0], embedding[:,1], c='maroon', marker='.', s=MARKER_SIZE)
    #     else:
    #         ax[i].scatter(embedding[:,0], embedding[:,1], embedding[:,2], marker='.', s=MARKER_SIZE, c='maroon')
    #     ax[i].set_xticks([])
    #     ax[i].set_yticks([])

    # fig.tight_layout(pad=0.5)
    # plt.savefig(f'{features}_features_umap_sub50-100.png')

    # if umap_dim == 3:
    #     fig, ax = plt.subplots(len(chan_dic.keys()), 3, sharey=False, subplot_kw=dict(projection="3d"), figsize=(6, 3*len(chan_dic.keys())))
    # elif umap_dim == 2:
    #     fig, ax = plt.subplots(len(chan_dic.keys()), 3, sharey=False, figsize=(6, 3*len(chan_dic.keys())))

    for i, channel in enumerate(chan_dic.keys()):
        fig, ax = plt.subplots(1, len(paths), sharey=False, figsize=(3*len(paths), 3))
        for j, path in enumerate(paths):
            print(f'{i}:{j} {channel}, {path}')
            f = np.load(features+'_'+channel+path, allow_pickle=True)
            if 'train' in path:
                y = np.load(f'data/{LABELED_DIR}/y_train.npy')
            elif 'valid' in path:
                y = np.load(f'data/{LABELED_DIR}/y_valid.npy')
            elif 'test' in path:
                y = np.load(f'data/{LABELED_DIR}/y_test.npy')
            else:
                print('I don\'t understand Python')
            
            if y.ndim > 1:
                y = np.argmax(y, axis=-1)

            print('Feature shape: ', f.shape)
            print('Label shape: ', y.shape)

            reducer = umap.UMAP(n_neighbors=umap_neighbors, n_components=umap_dim)
            embedding = reducer.fit_transform(f)
            if 'train' in path:    
                ax[j].set_title('Train')
            elif 'validation' in path:    
                ax[j].set_title('Validation')
            elif 'test' in path:    
                ax[j].set_title('Test')
            else:
                print('Unknown set type.')
            if umap_dim==2:
                ax[j].scatter(embedding[:,0], embedding[:,1], c=[color[i] for i in y], marker='.',s=MARKER_SIZE)
            else:
                ax[i][j].scatter(embedding[:,0], embedding[:,1], embedding[:,2], marker='.', c=[color[i] for i in y], s=MARKER_SIZE)
            ax[j].set_xticks([])
            ax[j].set_yticks([])
        fig.tight_layout(pad=0.5)
        plt.savefig(f'imgs/{features}_{channel}_features_umap_sub0-50.png')

    # fig.tight_layout(pad=0.5)
    # plt.savefig(f'{features}_features_umap_sub0-50.png')


