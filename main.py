import numpy as np
from hyperfast.hyper_network.loader import HyperNetworkLoader
from hyperfast.hyper_network.model import HyperNetworkGenerator

X_train, y_train = np.load("data/hapmap1_X_train.npy"), np.load( "data/hapmap1_y_train.npy")
X_test, y_test = np.load("data/hapmap1_X_test.npy"), np.load("data/hapmap1_y_test.npy")

network = HyperNetworkLoader.get_loaded_network()
hyper_network = HyperNetworkGenerator(network=network, n_ensemble=1)
trained = hyper_network.fit(X_train, y_train)