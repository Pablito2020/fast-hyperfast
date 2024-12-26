import numpy as np
from sklearn.metrics import accuracy_score

from hyperfast.hyper_network.model import HyperNetworkGenerator
from hyperfast.main_network.model import MainNetworkClassifier
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

    data_array = np.loadtxt("data/test.xls", delimiter=",", dtype=str)
    X_test = data_array[1:, :-1]  # All rows, all columns except the last
    y_test = data_array[1:, -1]   # All rows, only the last column
    X_test = np.array(X_train, dtype=np.number)
    y_test = np.array(y_train)
    return X_train, y_train, X_test, y_test


def get_original_ds():
    X_train, y_train = np.load("data/hapmap1_X_train.npy"), np.load( "data/hapmap1_y_train.npy")
    X_test, y_test = np.load("data/hapmap1_X_test.npy"), np.load("data/hapmap1_y_test.npy")
    return X_train, y_train, X_test, y_test


X_train, y_train, X_test, y_test = get_phone_ds()

hyper_network = HyperNetworkGenerator.load_from_pre_trained(n_ensemble=1)
classifier = hyper_network.generate_classifier_for_dataset(X_train, y_train)

predictions = classifier.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")

print(f"Fine tuning...")
classifier.fine_tune_networks(x=X_train, y=y_train, optimize_steps=64)
predictions = classifier.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")

print(f"Saving the model")
classifier.save_model(path="./model_cuda.pkl", device="cuda")

# An example of loading a model and predicting directly
# classifier = MainNetworkClassifier.load_from_pre_trained(path="./model_cuda.pkl")
# predictions = classifier.predict(X_test)
# accuracy = accuracy_score(y_test, predictions)
# print(f"Accuracy: {accuracy * 100:.2f}%")
