import math
from dataclasses import dataclass, field
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils import check_array, check_X_y, column_or_1d
from sklearn.utils.multiclass import check_classification_targets
from torch import Tensor

from hyperfast.hyper_network.configuration import DEFAULT_HYPER_NETWORK_CONFIGURATION


class TrainingDataConfig(BaseModel):
    batch_size: int = 2048
    stratify_sampling: bool = False
    feature_bagging: bool = False
    feature_bagging_size: int = 3000
    cat_features: List[int] = []


@dataclass(frozen=True)
class Transformers:
    """
    Transformers that we initialize with the test data. They will also allow us
    to transform the inference data!
    """

    one_hot_encoder: ColumnTransformer
    scaler: StandardScaler
    numerical_imputer: SimpleImputer | None = None
    categorical_imputer: SimpleImputer | None = None


class ProcessTrainingDataInformation(BaseModel):
    """
    Returns information about the dataset. Which classes it has, the selected_features, and
    the "transformers" => instances of objects that will allow us to transform the inference data.
    """

    classes: np.ndarray
    categorical_features: List[int] = []
    numerical_feature_ids: np.ndarray
    transformers: Transformers

    class Config:
        arbitrary_types_allowed = True


class ProcessorTrainingDataResult(BaseModel):
    data: Tuple[Tensor, Tensor]
    misc: ProcessTrainingDataInformation
    selected_features: List[Tensor] = []

    class Config:
        arbitrary_types_allowed = True


type Data = np.ndarray | pd.DataFrame


@dataclass
class TrainingDataProcessor:
    number_of_dimensions: int = DEFAULT_HYPER_NETWORK_CONFIGURATION.number_of_dimensions
    config: TrainingDataConfig = field(default_factory=TrainingDataConfig)

    def _assert_dataset_is_correct(self, x: Data, y: Data) -> Tuple[Data, Data]:
        if not isinstance(x, (np.ndarray, pd.DataFrame)) and not isinstance(
            y, (np.ndarray, pd.Series)
        ):
            x, y = check_X_y(x, y)
        if not isinstance(x, (np.ndarray, pd.DataFrame)):
            x = check_array(x)
        if not isinstance(y, (np.ndarray, pd.Series)):
            y = np.array(y)
        return np.array(x).copy(), np.array(y).copy()

    def _preprocess_categorical_features(
        self, x: Data
    ) -> Tuple[
        np.ndarray | pd.DataFrame,
        SimpleImputer,
        ColumnTransformer,
    ]:
        cat_imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
        cat_imputer.fit(x[:, self.config.cat_features])
        x[:, self.config.cat_features] = cat_imputer.transform(
            x[:, self.config.cat_features]
        )
        x = pd.DataFrame(x)
        one_hot_encoder = ColumnTransformer(
            transformers=[
                (
                    "cat",
                    OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                    self.config.cat_features,
                )
            ],
            remainder="passthrough",
        )
        one_hot_encoder.fit(x)
        x = one_hot_encoder.transform(x)
        return x, cat_imputer, one_hot_encoder

    def _preprocess_fitting_data(
        self,
        x: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
    ) -> ProcessorTrainingDataResult:
        x, y = self._assert_dataset_is_correct(x, y)

        if len(x.shape) == 2:
            _all_feature_idxs = np.arange(x.shape[1])
        else:
            raise ValueError("Reshape your data")

        numerical_feature_ids = np.setdiff1d(
            _all_feature_idxs, self.config.cat_features
        )
        if len(numerical_feature_ids) > 0:
            num_imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
            num_imputer.fit(x[:, numerical_feature_ids])
            x[:, numerical_feature_ids] = num_imputer.transform(
                x[:, numerical_feature_ids]
            )
        else:
            num_imputer = None

        if len(self.config.cat_features) > 0:
            x, cat_imputer, one_hot_encoder = self._preprocess_categorical_features(x)
        else:
            cat_imputer, one_hot_encoder = None, None

        x, y = check_X_y(x, y)
        # Standardize data
        scaler = StandardScaler()
        scaler.fit(x)
        x = scaler.transform(x)

        check_classification_targets(y)
        y = column_or_1d(y, warn=True)
        classes, y = np.unique(y, return_inverse=True)

        # Return what we've done
        transformers = Transformers(
            one_hot_encoder=one_hot_encoder,
            scaler=scaler,
            numerical_imputer=num_imputer,
            categorical_imputer=cat_imputer,
        )
        info = ProcessTrainingDataInformation(
            classes=classes,
            numerical_feature_ids=numerical_feature_ids,
            categorical_features=self.config.cat_features,
            transformers=transformers,
        )
        result = ProcessorTrainingDataResult(
            data=(
                torch.tensor(x, dtype=torch.float),
                torch.tensor(y, dtype=torch.long),
            ),
            misc=info,
        )
        return result

    def _sample_data(
        self, X: Tensor, y: Tensor
    ) -> Tuple[Tuple[Tensor, Tensor], List[Tensor]]:
        selected_features = []
        if self.config.feature_bagging:
            print("Performing feature bagging")
            stds = torch.std(X, dim=0)
            feature_idxs = torch.multinomial(
                stds, self.config.feature_bagging_size, replacement=False
            )
            selected_features.append(feature_idxs)
            X = X[:, feature_idxs]

        if self.config.stratify_sampling:
            print("Using stratified sampling")
            classes, class_counts = torch.unique(y, return_counts=True)
            samples_per_class = self.config.batch_size // len(classes)
            sampled_indices = []

            for cls in classes:
                cls_indices = (y == cls).nonzero(as_tuple=True)[0]
                n_samples = min(samples_per_class, len(cls_indices))
                cls_sampled_indices = cls_indices[
                    torch.randperm(len(cls_indices))[:n_samples]
                ]
                sampled_indices.append(cls_sampled_indices)

            sampled_indices = torch.cat(sampled_indices)
            sampled_indices = sampled_indices[torch.randperm(len(sampled_indices))]
        else:
            # Original random sampling
            sampled_indices = torch.randperm(len(X))[: self.config.batch_size]
        X_pred, y_pred = X[sampled_indices].flatten(start_dim=1), y[sampled_indices]
        if X_pred.shape[0] < self.number_of_dimensions:
            n_repeats = math.ceil(self.number_of_dimensions / X_pred.shape[0])
            X_pred = torch.repeat_interleave(X_pred, n_repeats, axis=0)
            y_pred = torch.repeat_interleave(y_pred, n_repeats, axis=0)
        return (X_pred, y_pred), selected_features

    def sample(self, x: Tensor, y: Tensor) -> ProcessorTrainingDataResult:
        res = self._preprocess_fitting_data(x, y)
        x, y = res.data
        result, selected_features = self._sample_data(x, y)
        return res.model_copy(
            update={"data": result, "selected_features": selected_features}
        )
