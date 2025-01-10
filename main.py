# %% [markdown]
# # Fast Hyperfast ⚡
# This notebook shows some of the uses of the library [fast_hyperfast](https://github.com/Pablito2020/fast-hyperfast/).
# fast_hyperfast tries to improve the original [hyperfast library](https://github.com/AI-sandbox/HyperFast) based on the 
# [hyperfast paper](https://arxiv.org/abs/2402.14335). It tries to improve it on the following aspects:
#  - Rewrote the code entirely to make an "easy to know what you are doing" library. You can easily see what you are training, infering, etc (in the original project you only have one sckit-learn like interface, which does all the "magic" for you)
#  - Allowing the user to save the main network weights
#  - Allowing the user to train the hypernetwork
# 
# In this notebook we'll show how we create a main network from a random dataset from kaggle, fine tune it and save his weights so we can load it on a low-level device

# %% [markdown]
# ## Installation
# We'll install [fast_hyperfast from pypy](https://pypi.org/project/fast_hyperfast/)
# If you want to see the code, [check it out on github](https://github.com/Pablito2020/fast-hyperfast/)

# %% 
# !pip install fast_hyperfast

# %% [markdown]
# ## Imports
# Import some utils from libraries, and the datasets from the /data folder

# %%
import numpy as np
from sklearn.metrics import accuracy_score

from hyperfast.hyper_network.model import HyperNetworkGenerator
from hyperfast.utils.seed import seed_everything

def get_phone_ds():
    """
    Get Mobile Price Classification dataset from:
    From: https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification?resource=download&select=train.csv
    """
    data_array = np.loadtxt("data/train.xls", delimiter=",", dtype=str)
    X_train = data_array[1:, :-1]  # All rows, all columns except the last
    y_train = data_array[1:, -1]  # All rows, only the last column
    X_train = np.array(X_train, dtype=np.number)
    y_train = np.array(y_train)

    data_array = np.loadtxt("data/test.xls", delimiter=",", dtype=str)
    X_test = data_array[1:, :-1]  # All rows, all columns except the last
    y_test = data_array[1:, -1]  # All rows, only the last column
    X_test = np.array(X_train, dtype=np.number)
    y_test = np.array(y_train)
    return X_train, y_train, X_test, y_test


def get_original_ds():
    X_train, y_train = (
        np.load("data/hapmap1_X_train.npy"),
        np.load("data/hapmap1_y_train.npy"),
    )
    X_test, y_test = (
        np.load("data/hapmap1_X_test.npy"),
        np.load("data/hapmap1_y_test.npy"),
    )
    return X_train, y_train, X_test, y_test

# %% [markdown]
# ## Seed everything, for reproducibility

# %%
seed_everything(seed=3)

# %% [markdown]
# ## Load the dataset
# %%
X_train, y_train, X_test, y_test = get_phone_ds()

# %% [markdown]
# ## Load the hypernetwork 
# Get the weights generated from the meta-training dataset that they give us on the paper.
# Then, generate a classifier (main network) given X_train
# We generate only one hypernetwork, and therefore we'll only generate one main network. In the paper they explain that they saw better results
# when using multiple ensembles (N hypernetworks that generate N main networks) and combine the results of the N main networks.
# %%
hyper_network = HyperNetworkGenerator.load_from_pre_trained(n_ensemble=1)
classifier = hyper_network.generate_classifier_for_dataset(X_train, y_train)

# %% [markdown]
# ## Out of the box performance
# Okay, now let's see how well it performs!
# %%
predictions = classifier.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")

# %% [markdown]
# ## Fine-Tunning the main network
# Maybe we can improve the accuracy if we fine tune the main network?
# (although 68.8 % for the first try wasn't bad at all...) Let's try it!
# %%
print("Fine tuning...")
classifier.fine_tune_networks(x=X_train, y=y_train, optimize_steps=64)
predictions = classifier.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")

# %% [markdown]
# ## Save the model
# Woah! That's pretty good! We can save the classifier to a .pkl file so we can run the 
# main network on another low-level device (without having to load the whole hypernetwork!)
# %%
print("Saving the model")
classifier.save_model(path="./model.pkl")

# %% [markdown]
# ## Load the model
# Now, given we have the weights of the classifier, we can load it on a less-intensive devices like, for example a raspberry pi 4.
# An example of loading a model and predicting directly:
# %%
from hyperfast.main_network.model import MainNetworkClassifier
classifier = MainNetworkClassifier.load_from_pre_trained(path="./model.pkl")
predictions = classifier.predict(X_test)

# We should have the same accuracy as before
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")
