import torch
from sklearn.decomposition import PCA
from torch import nn, Tensor

from hyperfast.utils.cuda import is_torch_pca
from hyperfast.utils.torch_pca import TorchPCA


class FixedSizeTransformer(nn.Module):

    def __init__(self, number_of_dimensions: int, random_feature_nn_size: int, clip_data_value: float):
        super().__init__()
        self.ndims = number_of_dimensions
        self.rf_size = random_feature_nn_size
        self.clip_data_value = clip_data_value

    def _get_random_features(self, X) -> Tensor:
        """
        Get Random Feature from X
        """
        rf_linear = nn.Linear(X.shape[1], self.rf_size, bias=False)
        nn.init.kaiming_normal_(rf_linear.weight, mode="fan_out", nonlinearity="relu")
        rf_linear.weight.requires_grad = False
        rf = nn.Sequential(rf_linear, nn.ReLU()).to(X.device)
        with torch.no_grad():
            return rf(X)

    def _get_pca(self, X) -> Tensor:
        pca = TorchPCA(n_components=self.ndims) if is_torch_pca() else PCA(n_components=self.ndims)
        if is_torch_pca():
            return pca.fit_transform(X)
        else:
            return torch.from_numpy(pca.fit_transform(X.cpu().numpy())).to(X.device)

    def _get_pca_mean_per_class(self, x, y, n_classes) -> Tensor:
        pca_perclass_mean = []
        for lab in range(n_classes):
            if torch.sum((y == lab)) > 0:
                class_mean = torch.mean(x[y == lab], dim=0, keepdim=True)
            else:
                class_mean = torch.mean(x, dim=0, keepdim=True)
            pca_perclass_mean.append(class_mean)
        return torch.cat(pca_perclass_mean)

    def transform(self, X) -> Tensor:
        """
        Transforms X to a tensor of n_dimensions, that has pca and random features taken into account.
        """
        X = X.flatten(start_dim=1)
        X = self._get_random_features(X)
        X = self._get_pca(X)
        return torch.clamp(X, -self.clip_data_value, self.clip_data_value)

    def get_pca_output(self, X, y, n_classes) -> Tensor:
        pca_global_mean = torch.mean(X, axis=0)
        pca_per_class_mean = self._get_pca_mean_per_class(x=X, y=y, n_classes=n_classes)

        pca_concat = []
        for ii, lab in enumerate(y):
            if pca_per_class_mean.ndim == 1:
                pca_per_class_mean = pca_per_class_mean.unsqueeze(0)
            if X.ndim == 1:
                X = X.unsqueeze(0)

            lab_index = lab.item() if torch.is_tensor(lab) else lab
            lab_index = min(lab_index, pca_per_class_mean.size(0) - 1)

            row = torch.cat((X[ii], pca_global_mean, pca_per_class_mean[lab_index]))
            pca_concat.append(row)
        return torch.vstack(pca_concat)