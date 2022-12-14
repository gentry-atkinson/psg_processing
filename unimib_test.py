import numpy as np
from model_wrappers import NNCLR_C
from utils.gen_ts_data import generate_pattern_data_as_array
import umap.umap_ as umap
from matplotlib import pyplot as plt
from data.twristar.load_data_time_series_dev.HAR.UniMiB_SHAR.unimib_shar_adl_load_dataset import unimib_load_dataset

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

K=5

if __name__ == '__main__':
    X_train, y_train,  X_val, y_val,  X_test, y_test = unimib_load_dataset(
        incl_xyz_accel = True,
        incl_rms_accel = False,
        incl_val_group = True,
        one_hot_encode=True
    )


    print(f'X_train: {X_train.shape}')
    print(f'X_val {X_val.shape}')
    print(f'X_test: {X_test.shape}')

    X_train = np.moveaxis(X_train, 2, 1)
    X_val = np.moveaxis(X_val, 2, 1)
    X_test = np.moveaxis(X_test, 2, 1)

    print(f'X_train after move axis: {X_train.shape}')
    print(f'X_val  after move axis: {X_val.shape}')
    print(f'X_test after move axis: {X_test.shape}')

    y_train = np.argmax(y_train, axis=-1)
    y_val = np.argmax(y_val, axis=-1)
    y_test = np.argmax(y_test, axis=-1)

    print(f'y_train: {y_train.shape}')
    print(f'y_val {y_val.shape}')
    print(f'y_test: {y_test.shape}')

    feature_learner = NNCLR_C(X=X_train, y=y_train)
    feature_learner.fit(
        X_train, y_train, X_val, y_val
    )

    f_train = feature_learner.get_features(X_train)
    f_val = feature_learner.get_features(X_val)
    f_test = feature_learner.get_features(X_test)

    print(f'f_train: {f_train.shape}')
    print(f'f_val {f_val.shape}')
    print(f'f_test: {f_test.shape}')

    #Print test picture

    #fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=False, subplot_kw=dict(projection="3d"))
    fig, ax = plt.subplots(1, 3, sharey=False)
    #ax = [ax1, ax2, ax3]

    reducer = umap.UMAP(n_neighbors=15, n_components=2)
    
    embedding = reducer.fit_transform(f_train)
    ax[0].set_title('Train')
    ax[0].scatter(embedding[:,0], embedding[:,1], marker='.', c=y_train)

    embedding = reducer.fit_transform(f_val)
    ax[1].set_title('Val')
    ax[1].scatter(embedding[:,0], embedding[:,1], marker='.', c=y_val)

    embedding = reducer.fit_transform(f_test)
    ax[2].set_title('Test')
    ax[2].scatter(embedding[:,0], embedding[:,1], marker='.', c=y_test)

    fig.tight_layout(pad=0.5)
    plt.savefig('unimib_data_train_val_test.png')

    neigh = KNeighborsClassifier(n_neighbors=K)
    neigh.fit(f_train, y_train)
    y_pred = neigh.predict(f_test)

    print(confusion_matrix(y_test, y_pred))
    print(f"KNN Accuracy: ", accuracy_score(y_test, y_pred))

