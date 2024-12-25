import numpy as np
import pandas as pd
import torch
from sklearn.utils import check_array
from torch import Tensor

# batch_size: int = 2048
# stratify_sampling: bool = False
# feature_bagging: bool = False
# feature_bagging_size: int = 3000
# cat_features: List[int] = []


class InferenceStandardizer:

    def _preprocess_test_data(
        self,
        x_test: np.ndarray | pd.DataFrame,
    ) -> Tensor:
        if not isinstance(x_test, (np.ndarray, pd.DataFrame)):
            x_test = check_array(x_test)
        x_test = np.array(x_test).copy()
        # Impute missing values for numerical features with the mean
        if len(x_test.shape) == 1:
            raise ValueError("Reshape your data")
        if len(self._numerical_feature_idxs) > 0:
            x_test[:, self._numerical_feature_idxs] = self._num_imputer.transform(
                x_test[:, self._numerical_feature_idxs]
            )

        if len(self._cat_features) > 0:
            # Impute missing values for categorical features with the most frequent category
            x_test[:, self._cat_features] = self.cat_imputer.transform(
                x_test[:, self._cat_features]
            )

            # One-hot encode categorical features
            x_test = pd.DataFrame(x_test)
            x_test = self.one_hot_encoder.transform(x_test)

        x_test = check_array(x_test)
        # Standardize data
        x_test = self._scaler.transform(x_test)
        return torch.tensor(x_test, dtype=torch.float)
