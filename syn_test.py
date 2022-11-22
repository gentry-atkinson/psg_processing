import numpy as np
from model_wrappers import NNCLR_C
from utils.gen_ts_data import generate_pattern_data_as_array
import umap.umap_ as umap
from matplotlib import pyplot as plt

def load_synthetic_dataset(incl_xyz_accel=True, incl_rms_accel=False, incl_val_group=True):
    NUM_TRAIN = 2001
    NUM_VAL = 201
    NUM_TEST = 201
    NUM_CLASSES = 2
    INSTANCE_LEN = 150

    params = {
        'avg_pattern_length' : [],
        'avg_amplitude' : [],
        'default_variance' : [],
        'variance_pattern_length' : [],
        'variance_amplitude' : []
    }

    for _ in range(NUM_CLASSES):
        params['avg_amplitude'].append(np.random.randint(0, 5))
        params['avg_pattern_length'].append(np.random.randint(5, 15))
        params['default_variance'].append(np.random.randint(1, 4))
        params['variance_pattern_length'].append(np.random.randint(5, 20))
        params['variance_amplitude'].append(np.random.randint(1, 5))

    train_set = np.zeros((NUM_TRAIN, INSTANCE_LEN))
    val_set = np.zeros((NUM_VAL, INSTANCE_LEN))
    test_set = np.zeros((NUM_TEST, INSTANCE_LEN))

    train_labels = []
    val_labels = []
    test_labels = []

    train_label_count = [0]*NUM_CLASSES
    val_label_count = [0]*NUM_CLASSES
    test_label_count = [0]*NUM_CLASSES

    for i in range (NUM_TRAIN):
        label = np.random.randint(0, NUM_CLASSES)
        # one_hot = np.zeros(NUM_CLASSES)
        # one_hot[label] = 1
        train_labels.append([0 if i!=label else 1 for i in range(NUM_CLASSES)])
        train_set[i, :] = generate_pattern_data_as_array(
            length=INSTANCE_LEN,
            avg_pattern_length=params['avg_pattern_length'][label],
            avg_amplitude=params['avg_amplitude'][label],
            default_variance=params['default_variance'][label],
            variance_pattern_length=params['variance_pattern_length'][label],
            variance_amplitude=params['variance_amplitude'][label]
        )
        train_label_count[label] += 1

    for i in range (NUM_VAL):
        label = np.random.randint(0, NUM_CLASSES)
        val_labels.append([0 if i!=label else 1 for i in range(NUM_CLASSES)])
        val_set[i, :] = generate_pattern_data_as_array(
            length=INSTANCE_LEN,
            avg_pattern_length=params['avg_pattern_length'][label],
            avg_amplitude=params['avg_amplitude'][label],
            default_variance=params['default_variance'][label],
            variance_pattern_length=params['variance_pattern_length'][label],
            variance_amplitude=params['variance_amplitude'][label]
        )
        val_label_count[label] += 1

    for i in range (NUM_VAL):
        label = np.random.randint(0, NUM_CLASSES)
        test_labels.append([0 if i!=label else 1 for i in range(NUM_CLASSES)])
        test_set[i, :] = generate_pattern_data_as_array(
            length=INSTANCE_LEN,
            avg_pattern_length=params['avg_pattern_length'][label],
            avg_amplitude=params['avg_amplitude'][label],
            default_variance=params['default_variance'][label],
            variance_pattern_length=params['variance_pattern_length'][label],
            variance_amplitude=params['variance_amplitude'][label]
        )
        test_label_count[label] += 1


    train_set = np.reshape(train_set, (train_set.shape[0], train_set.shape[1], 1))
    val_set = np.reshape(val_set, (val_set.shape[0], val_set.shape[1], 1))
    test_set = np.reshape(test_set, (test_set.shape[0], test_set.shape[1], 1))

    train_labels = np.array(train_labels)
    val_labels = np.array(val_labels)
    test_labels = np.array(test_labels)

    print("Train labels: ", '\n'.join([str(i) for i in train_label_count]))
    print("Validatoions labels: ", '\n'.join([str(i) for i in val_label_count]))
    print("Test labels: ", '\n'.join([str(i) for i in test_label_count]))

    print("Train data shape: ", train_set.shape)
    print("Validation data shape: ", val_set.shape)
    print("Test data shape: ", test_set.shape)

    return train_set, train_labels, val_set, val_labels, test_set, test_labels

if __name__ == '__main__':
    X_full_train, y_full_train,  X_full_val, y_full_val,  X_full_test, y_full_test = load_synthetic_dataset()

    X_first_train = X_full_train[0:len(X_full_train)//2]
    X_first_val = X_full_train[0:len(X_full_val)//2]
    X_first_test = X_full_train[0:len(X_full_test)//2]

    y_train = X_full_train[0:len(y_full_train)//2]
    y_val = X_full_train[0:len(y_full_val)//2]
    y_test = X_full_train[0:len(y_full_test)//2]

    X_second_train = X_full_train[len(X_full_train)//2:]
    X_second_val = X_full_train[len(X_full_val)//2:]
    X_second_test = X_full_train[len(X_full_test)//2:]

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
