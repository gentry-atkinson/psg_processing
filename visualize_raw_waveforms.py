import numpy as np
import random
from matplotlib import pyplot as plt
import os
from sklearn.model_selection import train_test_split

#FOR PSG AUDIO
chan_dic = {
    'Thermistor Flow': [9],
    'Respiratory Belt together' : [10, 11],
    'Thorasis': [10],
    'Abdominal': [11],
    'ECG' : [8],
    'EOG' : [3, 4],
    'EMG' : [5],
    'Leg' : [6, 7]
} 

#FOR TWRISTAR
# chan_dic = {
#     'Accel' : [0],
#     'BVP' : [1],
#     'EDA' : [2],
#     'P Temp' : [3],
#     'All together' : [0, 1, 2, 3]
# }

# UNLABELED_DIR = 'twristar/unlabeled'
# LABELED_DIR = 'twristar/labeled'

UNLABELED_DIR = 'second 50'
LABELED_DIR = 'first 50'

color = ['maroon', 'darkblue', 'gold', 'green']

if __name__ == '__main__':

    axes = []
    fig, axes = plt.subplots(len(chan_dic), 1, sharey=False)

    #data is channels last

    print('Reading (unlabeled) subjects 50-100...')
    x_train_second = np.load(f'data/{UNLABELED_DIR}/x_train.npy', allow_pickle=True)

    if os.path.exists(f'data/{UNLABELED_DIR}/x_valid.npy'):
        x_test_second = np.load(f'data/{UNLABELED_DIR}/x_test.npy', allow_pickle=True)
    else:
        x_train_second, x_test_second = train_test_split( x_train_second, test_size=0.1, shuffle=False)

    if os.path.exists(f'data/{UNLABELED_DIR}/x_valid.npy'):
        x_val_second = np.load(f'data/{UNLABELED_DIR}/x_valid.npy', allow_pickle=True)
    else:
        x_train_second, x_val_second = train_test_split( x_train_second, test_size=0.1, shuffle=False)

    print('Second X test shape: ', x_test_second.shape)
    print('Second X val shape: ', x_val_second.shape)
    print('Second X train shape: ', x_train_second.shape)

    print('Reading (labeled) subjects 0-49...')
    x_train_first = np.load(f'data/{LABELED_DIR}/x_train.npy', allow_pickle=True)
    x_test_first = np.load(f'data/{LABELED_DIR}/x_test.npy', allow_pickle=True)
    if os.path.exists(f'data/{LABELED_DIR}/x_valid.npy'):
        x_val_first = np.load(f'data/{LABELED_DIR}/x_valid.npy', allow_pickle=True)
    else:
        x_train_first, x_val_first = train_test_split(x_train_first, test_size=0.1, shuffle=False)

    data = x_train_first
    data_str = 'x_train_first'

    instance_to_plot = random.randint(0, data.shape[0]-1)
    fig.suptitle(f'Instance: {instance_to_plot} of {data_str}')

    for i, k in enumerate(chan_dic.keys()):
        #axes[i%3][i//3].set_title(k)
        axes[i].set_title(k)
        for j, c in enumerate(chan_dic[k]):
            #axes[i%3][i//3].plot(range(data.shape[1]), data[instance_to_plot, :, c], c=color[j])
            axes[i].plot(range(data.shape[1]), data[instance_to_plot, :, c], c=color[j])
        #fig.title

    fig.tight_layout()
    #plt.show()
    plt.savefig(f'raw psg visualizations/visualize_raw_{data_str}.pdf')



