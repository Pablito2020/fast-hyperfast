from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils import column_or_1d
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_X_y, check_array
from torch import Tensor

from hyperfast.hyper_network.network import HyperNetwork
from hyperfast.main_network.model import MainNetworkModel


class HyperNetworkGenerator(BaseEstimator, ClassifierMixin):
    def __init__(self, network: HyperNetwork, n_ensemble: int = 16, batch_size: int = 2048,
                 stratify_sampling: bool = False, feature_bagging: bool = False, feature_bagging_size: int = 3000,
                 cat_features: List[int] | None = None, ) -> None:
        self.n_ensemble = n_ensemble
        self.batch_size = batch_size
        self.stratify_sampling = stratify_sampling
        self.feature_bagging = feature_bagging
        self.feature_bagging_size = feature_bagging_size
        self.cat_features = cat_features
        self.selected_features = []
        # TODO: This could be better
        self._model = network
        self.configuration = network.config
        self.device = network.loaded_on

    def _get_tags(self) -> dict:
        tags = super()._get_tags()
        tags["allow_nan"] = True
        return tags

    def _preprocess_fitting_data(self, x: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series, ) -> Tuple[
        Tensor, Tensor]:
        if not isinstance(x, (np.ndarray, pd.DataFrame)) and not isinstance(y, (np.ndarray, pd.Series)):
            x, y = check_X_y(x, y)
        if not isinstance(x, (np.ndarray, pd.DataFrame)):
            x = check_array(x)
        if not isinstance(y, (np.ndarray, pd.Series)):
            y = np.array(y)
        x = np.array(x).copy()
        y = np.array(y).copy()
        self._cat_features = self.cat_features if self.cat_features is not None else []
        # Impute missing values for numerical features with the mean
        self._num_imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
        if len(x.shape) == 2:
            self._all_feature_idxs = np.arange(x.shape[1])
        else:
            raise ValueError("Reshape your data")
        self._numerical_feature_idxs = np.setdiff1d(self._all_feature_idxs, self._cat_features)
        if len(self._numerical_feature_idxs) > 0:
            self._num_imputer.fit(x[:, self._numerical_feature_idxs])
            x[:, self._numerical_feature_idxs] = self._num_imputer.transform(x[:, self._numerical_feature_idxs])

        if len(self._cat_features) > 0:
            # Impute missing values for categorical features with the most frequent category
            self.cat_imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
            self.cat_imputer.fit(x[:, self._cat_features])
            x[:, self._cat_features] = self.cat_imputer.transform(x[:, self._cat_features])

            # One-hot encode categorical features
            x = pd.DataFrame(x)
            self.one_hot_encoder = ColumnTransformer(transformers=[
                ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), self._cat_features,)],
                remainder="passthrough", )
            self.one_hot_encoder.fit(x)
            x = self.one_hot_encoder.transform(x)

        x, y = check_X_y(x, y)
        # Standardize data
        self._scaler = StandardScaler()
        self._scaler.fit(x)
        x = self._scaler.transform(x)

        check_classification_targets(y)
        y = column_or_1d(y, warn=True)
        self.n_features_in_ = x.shape[1]
        self.classes_, y = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        return torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.long)

    def _preprocess_test_data(self, x_test: np.ndarray | pd.DataFrame, ) -> Tensor:
        if not isinstance(x_test, (np.ndarray, pd.DataFrame)):
            x_test = check_array(x_test)
        x_test = np.array(x_test).copy()
        # Impute missing values for numerical features with the mean
        if len(x_test.shape) == 1:
            raise ValueError("Reshape your data")
        if len(self._numerical_feature_idxs) > 0:
            x_test[:, self._numerical_feature_idxs] = self._num_imputer.transform(
                x_test[:, self._numerical_feature_idxs])

        if len(self._cat_features) > 0:
            # Impute missing values for categorical features with the most frequent category
            x_test[:, self._cat_features] = self.cat_imputer.transform(x_test[:, self._cat_features])

            # One-hot encode categorical features
            x_test = pd.DataFrame(x_test)
            x_test = self.one_hot_encoder.transform(x_test)

        x_test = check_array(x_test)
        # Standardize data
        x_test = self._scaler.transform(x_test)
        return torch.tensor(x_test, dtype=torch.float)

    def _sample_data(self, X: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        if self.feature_bagging:
            print("Performing feature bagging")
            stds = torch.std(X, dim=0)
            feature_idxs = torch.multinomial(stds, self.feature_bagging_size, replacement=False)
            self.selected_features.append(feature_idxs)
            X = X[:, feature_idxs]

        if self.stratify_sampling:
            # Stratified sampling
            print("Using stratified sampling")
            classes, class_counts = torch.unique(y, return_counts=True)
            samples_per_class = self.batch_size // len(classes)
            sampled_indices = []

            for cls in classes:
                cls_indices = (y == cls).nonzero(as_tuple=True)[0]
                n_samples = min(samples_per_class, len(cls_indices))
                cls_sampled_indices = cls_indices[torch.randperm(len(cls_indices))[:n_samples]]
                sampled_indices.append(cls_sampled_indices)

            sampled_indices = torch.cat(sampled_indices)
            sampled_indices = sampled_indices[torch.randperm(len(sampled_indices))]
        else:
            # Original random sampling
            sampled_indices = torch.randperm(len(X))[: self.batch_size]
        X_pred, y_pred = X[sampled_indices].flatten(start_dim=1), y[sampled_indices]
        if X_pred.shape[0] < self.configuration.number_of_dimensions:
            n_repeats = math.ceil(self.configuration.number_of_dimensions / X_pred.shape[0])
            X_pred = torch.repeat_interleave(X_pred, n_repeats, axis=0)
            y_pred = torch.repeat_interleave(y_pred, n_repeats, axis=0)
        return X_pred, y_pred

    def fit(self, x: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series) -> MainNetworkModel:
        """
        Generates a main model for the given data.

        Args:
            x (array-like): Input features.
            y (array-like): Target values.
        """
        _x, _y = self._preprocess_fitting_data(x, y)

        models = []
        for n in range(self.n_ensemble):
            X_pred, y_pred = self._sample_data(_x, _y)
            X_pred, y_pred = X_pred.to(self.device), y_pred.to(self.device)
            with torch.no_grad():
                models.append(self._model(X_pred, y_pred, self.n_classes_))
        return MainNetworkModel(models=models)

    def predict(self, _: any):
        raise NotImplementedError(
            "Predict shouldn't be used on the hyper network! You get the trained network via the .fit method!")
