from typing import Tuple

import torch
from sklearn.decomposition import PCA
from torch import nn, Tensor

from hyperfast.hyper_network.configuration import DEFAULT_RANDOM_FEATURE_SIZE
from hyperfast.utils.cuda import is_torch_pca


class RandomFeatures(nn.Module):
    def __init__(
        self, input_shape: int, random_feature_size: int = DEFAULT_RANDOM_FEATURE_SIZE
    ):
        super().__init__()
        rf_linear = nn.Linear(input_shape, random_feature_size, bias=False)
        nn.init.kaiming_normal_(rf_linear.weight, mode="fan_out", nonlinearity="relu")
        rf_linear.weight.requires_grad = False
        self.random_feature = nn.Sequential(rf_linear, nn.ReLU())

    def forward(self, x: Tensor) -> Tensor:
        return self.random_feature(x)


def svd_flip(u, v, u_based_decision=True):
    """Sign correction to ensure deterministic output from SVD.

    Adjusts the columns of u and the rows of v such that the loadings in the
    columns in u that are largest in absolute value are always positive.
    """
    if u_based_decision:
        # columns of u, rows of v
        max_abs_cols = torch.argmax(torch.abs(u), axis=0)
        signs = torch.sign(u[max_abs_cols, range(u.shape[1])])
        u *= signs
        v *= signs[:, None]
    else:
        # rows of v, columns of u
        max_abs_rows = torch.argmax(torch.abs(v), axis=1)
        signs = torch.sign(v[range(v.shape[0]), max_abs_rows])
        u *= signs
        v *= signs[:, None]
    return u, v


class TorchPCA:
    def __init__(self, n_components=None, fit="full"):
        self.n_components = n_components
        self.fit = fit

    def _fit(self, X: Tensor):
        if self.n_components is None:
            n_components = min(X.shape)
        else:
            n_components = self.n_components

        n_samples, n_features = X.shape
        if n_components > min(X.shape):
            raise ValueError(
                f"n_components should be <= min(n_samples: {n_samples}, n_features: {n_features})"
            )

        self.mean_ = torch.mean(X, axis=0)
        X -= self.mean_

        if self.fit == "full":
            U, S, Vt = torch.linalg.svd(X, full_matrices=False)
            # flip eigenvectors' sign to enforce deterministic output
            U, Vt = svd_flip(U, Vt)
        elif self.fit == "lowrank":
            U, S, Vt = torch.pca_lowrank(X)

        self.components_ = Vt[:n_components]
        self.n_components_ = n_components

        return U, S, Vt

    def fit(self, X: Tensor):
        self._fit(X)
        return self

    def transform(self, X: Tensor):
        assert self.mean_ is not None
        X -= self.mean_
        return torch.matmul(X, self.components_.T)

    def fit_transform(self, X: Tensor):
        U, S, Vt = self._fit(X)
        U = U[:, : self.n_components_]
        U *= S[: self.n_components_]
        return U


def get_pca(X: Tensor, number_dimensions: int) -> Tuple[Tensor, any]:
    pca = (
        TorchPCA(n_components=number_dimensions)
        if is_torch_pca()
        else PCA(n_components=number_dimensions)
    )
    if is_torch_pca():
        res = pca.fit_transform(X)
    else:
        res = torch.from_numpy(pca.fit_transform(X.cpu().numpy())).to(X.device)
    return res, pca


def _get_mean_per_class(x: Tensor, y: Tensor, n_classes: int) -> Tensor:
    mean_per_class = []
    for class_num in range(n_classes):
        if torch.sum((y == class_num)) > 0:
            class_mean = torch.mean(x[y == class_num], dim=0, keepdim=True)
        else:
            class_mean = torch.mean(x, dim=0, keepdim=True)
        mean_per_class.append(class_mean)
    return torch.cat(mean_per_class)


def get_mean_per_class(x: Tensor, y: Tensor, n_classes: int) -> Tensor:
    global_mean = torch.mean(input=x, axis=0)
    mean_per_class = _get_mean_per_class(x=x, y=y, n_classes=n_classes)

    # TODO: This transformations were living inside the original codebase, are they really necessary?
    # if mean_per_class.ndim == 1:
    #     mean_per_class = mean_per_class.unsqueeze(0)
    # if X.ndim == 1:
    #     X = X.unsqueeze(0)
    pca_concat = []
    for current_row, value_to_infer in enumerate(y):
        class_index = (
            value_to_infer.item() if torch.is_tensor(value_to_infer) else value_to_infer
        )
        assert (
            class_index <= mean_per_class.size(0) - 1
        ), "Is impossible that the index is bigger than the classes size!"
        row = torch.cat((x[current_row], global_mean, mean_per_class[class_index]))
        pca_concat.append(row)
    return torch.vstack(pca_concat)
