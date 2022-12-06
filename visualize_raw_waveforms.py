import numpy as np
import random
from matplotlib import pyplot as plt

chan_dic = {
    'Thermistor Flow': [9],
    'Respiratory Belt' : [10, 11],
    'ECG' : [8],
    'EOG' : [3, 4],
    'EMG' : [5],
    'Leg' : [6, 7]
}

UNLABELED_DIR = 'second 50'
LABELED_DIR = 'first 50'

color = ['maroon', 'darkblue']

if __name__ == '__main__':

    axes = []
    fig, axes = plt.subplots(3, 2, sharey=False)

    #data is channels last

    print('Reading subjects 50-100...')
    x_train_second = np.load(f'data/{UNLABELED_DIR}/x_train.npy', allow_pickle=True)
    x_val_second = np.load(f'data/{UNLABELED_DIR}/x_valid.npy', allow_pickle=True)
    x_test_second = np.load(f'data/{UNLABELED_DIR}/x_test.npy', allow_pickle=True)

    print('Second X test shape: ', x_test_second.shape)
    print('Second X val shape: ', x_val_second.shape)
    print('Second X train shape: ', x_train_second.shape)

    print('Reading subjects 0-49...')
    x_train_first = np.load(f'data/{LABELED_DIR}/x_train.npy', allow_pickle=True)
    x_val_first = np.load(f'data/{LABELED_DIR}/x_valid.npy', allow_pickle=True)
    x_test_first = np.load(f'data/{LABELED_DIR}/x_test.npy', allow_pickle=True)

    instance_to_plot = random.randint(0, x_train_second.shape[0]-1)
    fig.suptitle(f'Instance: {instance_to_plot}')

    for i, k in enumerate(chan_dic.keys()):
        axes[i%3][i//3].set_title(k)
        for j, c in enumerate(chan_dic[k]):
            axes[i%3][i//3].plot(range(x_train_second.shape[1]), x_train_second[instance_to_plot, :, c], c=color[j])
        #fig.title

    fig.tight_layout()
    plt.show()



