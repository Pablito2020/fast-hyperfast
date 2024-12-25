from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.utils import check_array
from torch import Tensor

from hyperfast.standardize_data.training import ProcessorTrainingDataResult


@dataclass(frozen=True)
class InferenceStandardizer:
    data: ProcessorTrainingDataResult

    def preprocess_inference_data(
        self,
        x_test: np.ndarray | pd.DataFrame,
    ) -> Tensor:
        # Assertions
        if not isinstance(x_test, (np.ndarray, pd.DataFrame)):
            x_test = check_array(x_test)
        x_test = np.array(x_test).copy()
        if len(x_test.shape) == 1:
            raise ValueError("Reshape your data")

        # Numerical
        numerical_feature_ids = self.data.misc.numerical_feature_ids
        if len(numerical_feature_ids) > 0:
            x_test[:, numerical_feature_ids] = self.data.misc.transformers.numerical_imputer.transform(
                x_test[:, numerical_feature_ids]
            )

        # Categorical
        cat_features = self.data.misc.categorical_features
        if len(cat_features) > 0:
            x_test[:, cat_features] = self.data.misc.transformers.categorical_imputer.transform(
                x_test[:, cat_features]
            )
            x_test = pd.DataFrame(x_test)
            x_test = self.data.misc.transformers.one_hot_encoder.transform(x_test)

        x_test = check_array(x_test)
        # Standardize data
        x_test = self.data.misc.transformers.scaler.transform(x_test)
        return torch.tensor(x_test, dtype=torch.float)
