import numpy as np
from hyperfast.hyper_network.loader import HyperNetworkLoader
from hyperfast.hyper_network.model import HyperNetworkGenerator
from hyperfast.utils.seed import seed_everything

seed_everything(seed=3)


def get_phone_ds():
    """
    Get Mobile Price Classification dataset from:
    From: https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification?resource=download&select=train.csv
    """
    data_array = np.loadtxt("data/train.xls", delimiter=",", dtype=str)
    X_train = data_array[1:, :-1]  # All rows, all columns except the last
    y_train = data_array[1:, -1]   # All rows, only the last column
    X_train = np.array(X_train, dtype=np.number)
    y_train = np.array(y_train)
    return X_train, y_train


def get_original_ds():
    X_train, y_train = np.load("data/hapmap1_X_train.npy"), np.load( "data/hapmap1_y_train.npy")
    X_test, y_test = np.load("data/hapmap1_X_test.npy"), np.load("data/hapmap1_y_test.npy")
    return X_train, y_train, X_test, y_test


# X_train, y_train, X_test, y_test = get_original_ds()
X_train, y_train = get_phone_ds()
network = HyperNetworkLoader.get_loaded_network()
hyper_network = HyperNetworkGenerator(network=network, n_ensemble=1)
main_network = hyper_network.generate_main_network_from_dataset(X_train, y_train)
print(main_network)
