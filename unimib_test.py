import numpy as np
from model_wrappers import NNCLR_C
from utils.gen_ts_data import generate_pattern_data_as_array
import umap.umap_ as umap
from matplotlib import pyplot as plt
from data.twristar.load_data_time_series_dev.HAR.UniMiB_SHAR import unimib_shar_adl_load_dataset

if __name__ == '__main__':
    X_full_train, y_full_train,  X_full_val, y_full_val,  X_full_test, y_full_test = load_synthetic_dataset()

    X_first_train = X_full_train[0:len(X_full_train)//2]
    X_first_val = X_full_val[0:len(X_full_val)//2]
    X_first_test = X_full_test[0:len(X_full_test)//2]

    y_train = y_full_train[0:len(y_full_train)//2]
    y_val = y_full_val[0:len(y_full_val)//2]
    y_test = y_full_test[0:len(y_full_test)//2]

    X_second_train = X_full_train[len(X_full_train)//2:]
    X_second_val = X_full_val[len(X_full_val)//2:]
    X_second_test = X_full_test[len(X_full_test)//2:]

    X_all = np.concatenate((X_second_train, X_second_val), axis=0)

    X_all = np.moveaxis(X_all, 2, 1)
    X_second_test = np.moveaxis(X_second_test, 2, 1)

    print(f'X_second_train: {X_second_train.shape}')
    print(f'X_second_val {X_second_val.shape}')
    print(f'X_second_test: {X_second_test.shape}')
    print(f'X_all: {X_all.shape}')

    X_first_train = np.moveaxis(X_first_train, 2, 1)
    X_first_val = np.moveaxis(X_first_val, 2, 1)
    X_first_test = np.moveaxis(X_first_test, 2, 1)

    print(f'X_first_train: {X_first_train.shape}')
    print(f'X_first_val {X_first_val.shape}')
    print(f'X_first_test: {X_first_test.shape}')

    y_train = np.argmax(y_train, axis=-1)
    y_val = np.argmax(y_val, axis=-1)
    y_test = np.argmax(y_test, axis=-1)

    print(f'y_train: {y_train.shape}')
    print(f'y_val {y_val.shape}')
    print(f'y_test: {y_test.shape}')

    feature_learner = NNCLR_C(X=X_all, y=[1])
    feature_learner.fit(
        X_all[:,[0],:], np.ones(X_all.shape[0]), X_second_test[:,[0],:], np.ones(X_second_test.shape[0])
    )

    f_train = feature_learner.get_features(X_first_train)
    f_val = feature_learner.get_features(X_first_val)
    f_test = feature_learner.get_features(X_first_test)

    print(f'f_train: {f_train.shape}')
    print(f'f_val {f_val.shape}')
    print(f'f_test: {f_test.shape}')

    #Print test picture

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=False, subplot_kw=dict(projection="3d"))
    ax = [ax1, ax2, ax3]

    reducer = umap.UMAP(n_neighbors=15, n_components=3)
    
    embedding = reducer.fit_transform(f_train)
    ax[0].set_title('Train')
    ax[0].scatter(embedding[:,0], embedding[:,1], embedding[:,2], marker='.', c=y_train)

    embedding = reducer.fit_transform(f_val)
    ax[1].set_title('Val')
    ax[1].scatter(embedding[:,0], embedding[:,1], embedding[:,2], marker='.', c=y_val)

    embedding = reducer.fit_transform(f_test)
    ax[2].set_title('Test')
    ax[2].scatter(embedding[:,0], embedding[:,1], embedding[:,2], marker='.', c=y_test)


    plt.savefig('synthetic_data_train_val_test.png')
