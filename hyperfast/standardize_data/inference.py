from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils import check_array
from torch import Tensor

from hyperfast.standardize_data.training import Transformers

class MyTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, numerical_feature_ids: np.ndarray, categorical_features: List[int], transformers: Transformers):
        self.numerical_feature_ids = numerical_feature_ids
        self.categorical_features = categorical_features
        self.transformers = transformers

    def fit(self, x_test, y=None):
        if not isinstance(x_test, (np.ndarray, pd.DataFrame)):
            x_test = check_array(x_test)
        x_test = np.array(x_test).copy()
        if len(x_test.shape) == 1:
            raise ValueError("Reshape your data")
        return x_test

    def transform(self, x_test):
        # Numerical
        numerical_feature_ids = self.numerical_feature_ids
        if len(numerical_feature_ids) > 0:
            x_test[:, numerical_feature_ids] = self.transformers.numerical_imputer.transform(
                x_test[:, numerical_feature_ids]
            )
        # Categorical
        cat_features = self.categorical_features
        if len(cat_features) > 0:
            x_test[:, cat_features] = self.transformers.categorical_imputer.transform(
                x_test[:, cat_features]
            )
            x_test = pd.DataFrame(x_test)
            x_test = self.transformers.one_hot_encoder.transform(x_test)
        return self.transformers.scaler.transform(x_test)


@dataclass(frozen=True)
class InferenceStandardizer:
    pipeline: Pipeline

    @staticmethod
    def from_training_data(numerical_feature_ids: np.ndarray, categorical_features: List[int], transformers: Transformers) -> InferenceStandardizer:
        pipeline = Pipeline(
            steps = [
                ("transform_data", MyTransformer(numerical_feature_ids, categorical_features, transformers)),
            ]
        )
        return InferenceStandardizer(pipeline=pipeline)

    @staticmethod
    def __assert_correct_input_data(x: np.ndarray | pd.DataFrame) -> np.ndarray:
        if not isinstance(x, (np.ndarray, pd.DataFrame)):
            x = check_array(x)
        x = np.array(x).copy()
        if len(x.shape) == 1:
            raise ValueError("Reshape your data")
        return x


    def save(self, path: str):
        joblib.dump(self.pipeline, path)

    @staticmethod
    def from_pre_trained(path: str) -> InferenceStandardizer:
        pipeline = joblib.load(path)
        return InferenceStandardizer(pipeline=pipeline)

    def preprocess_inference_data(
        self,
        x_test: np.ndarray | pd.DataFrame,
    ) -> Tensor:
        x_test = InferenceStandardizer.__assert_correct_input_data(x_test)
        x_test = self.pipeline.transform(x_test)
        x_test = check_array(x_test)
        return torch.tensor(x_test, dtype=torch.float)
