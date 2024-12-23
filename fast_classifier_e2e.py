import torch
import numpy as np
from hyperfast import HyperFastClassifier
from sklearn.metrics import accuracy_score

# Load your dataset
X_train, y_train = np.load("data/hapmap1_X_train.npy"), np.load( "data/hapmap1_y_train.npy")
X_test, y_test = np.load("data/hapmap1_X_test.npy"), np.load("data/hapmap1_y_test.npy")

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = "cpu"
print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
print(f"Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

# Initialize HyperFast
model = HyperFastClassifier(device=device, n_ensemble=1, optimization=None, seed=42)

# Generate a target network and make predictions
model.fit(X_train, y_train)
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")