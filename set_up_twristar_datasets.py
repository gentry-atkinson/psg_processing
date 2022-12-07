import numpy as np
from sklearn.model_selection import train_test_split

from data.twristar.load_data_time_series_dev.HAR.e4_wristband_Nov2019.e4_load_dataset import e4_load_dataset
#return x_train, y_train, x_validation, y_validation, x_test, y_test

UNLABELED_DIR = 'data/twristar/unlabeled'
LABELED_DIR = 'data/twristar/labeled'

if __name__ == '__main__':
    x_unlabeled_all = np.load (f'{UNLABELED_DIR}/original/x_train.npy')
    y_unlabeled_all = np.load (f'{UNLABELED_DIR}/original/y_train.npy')

    x_unlabeled_train, x_unlabeled_test, y_unlabeled_train, y_unlabeled_test = train_test_split(
        x_unlabeled_all, y_unlabeled_all, test_size=0.1)
    x_unlabeled_train, x_unlabeled_val, y_unlabeled_train, y_unlabeled_val = train_test_split(
        x_unlabeled_train, y_unlabeled_train, test_size=0.1, shuffle=False)

    x_labeled_train, y_labeled_train, x_labeled_val, y_labeled_val, x_labeled_test, y_labeled_test = e4_load_dataset(
        incl_val_group=True, incl_xyz_accel=True, incl_rms_accel=True
    )

    print('X unlabeled train shape: ', x_unlabeled_train.shape)
    print('y unlabeled train shape: ', y_unlabeled_train.shape)
    print('X unlabeled val shape: ', x_unlabeled_val.shape)
    print('y unlabeled val shape: ', y_unlabeled_val.shape)
    print('X unlabeled test shape: ', x_unlabeled_test.shape)
    print('y unlabeled test shape: ', y_unlabeled_test.shape)

    print('X labeled train shape: ', x_labeled_train.shape)
    print('y labeled train shape: ', y_labeled_train.shape)
    print('X labeled val shape: ', x_labeled_val.shape)
    print('y labeled val shape: ', y_labeled_val.shape)
    print('X labeled test shape: ', x_labeled_test.shape)
    print('y labeled test shape: ', y_labeled_test.shape)

    #PLEASE EXPLAIN THE SHAPE OF THE 'UNLABELED' Y VALUES

    np.save(f'{UNLABELED_DIR}/x_train.npy', x_unlabeled_train)
    np.save(f'{UNLABELED_DIR}/x_valid.npy', x_unlabeled_val)
    np.save(f'{UNLABELED_DIR}/x_test.npy', x_unlabeled_test)

    np.save(f'{LABELED_DIR}/x_train.npy', x_labeled_train)
    np.save(f'{LABELED_DIR}/x_valid.npy', x_labeled_val)
    np.save(f'{LABELED_DIR}/x_test.npy', x_labeled_test)

    np.save(f'{LABELED_DIR}/y_train.npy', y_labeled_train)
    np.save(f'{LABELED_DIR}/y_valid.npy', y_labeled_val)
    np.save(f'{LABELED_DIR}/y_test.npy', y_labeled_test)
    


