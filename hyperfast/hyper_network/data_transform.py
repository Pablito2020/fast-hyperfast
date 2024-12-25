import torch
from sklearn.decomposition import PCA
from torch import nn, Tensor

from hyperfast.utils.cuda import is_torch_pca
from hyperfast.utils.torch_pca import TorchPCA

DEFAULT_CLIP_DATA_VALUE = 27.6041
DEFAULT_RANDOM_FEATURE_SIZE =  2**15

class RandomFeatures(nn.Module):

    def __init__(self, input_shape: int, random_feature_size: int = DEFAULT_RANDOM_FEATURE_SIZE):
        super().__init__()
        rf_linear = nn.Linear(input_shape, random_feature_size, bias=False)
        nn.init.kaiming_normal_(rf_linear.weight, mode="fan_out", nonlinearity="relu")
        rf_linear.weight.requires_grad = False
        self.random_feature = nn.Sequential(rf_linear, nn.ReLU())

    def forward(self, x: Tensor) -> Tensor:
        return self.random_feature(x)



class FixedSizeTransformer:

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

    def transform(self, X) -> Tensor:
        """
        Transforms X to a tensor of n_dimensions, that has pca and random features taken into account.
        """
        X = X.flatten(start_dim=1)
        X = self._get_random_features(X)
        X = self._get_pca(X)
        return torch.clamp(X, -self.clip_data_value, self.clip_data_value)

    def _get_mean_per_class(self, x: Tensor, y: Tensor, n_classes: int) -> Tensor:
        mean_per_class = []
        for class_num in range(n_classes):
            if torch.sum((y == class_num)) > 0:
                class_mean = torch.mean(x[y == class_num], dim=0, keepdim=True)
            else:
                class_mean = torch.mean(x, dim=0, keepdim=True)
            mean_per_class.append(class_mean)
        return torch.cat(mean_per_class)


    def get_mean_per_class(self, x: Tensor, y: Tensor, n_classes: int) -> Tensor:
        global_mean = torch.mean(input=x, axis=0)
        mean_per_class = self._get_mean_per_class(x=x, y=y, n_classes=n_classes)

        # TODO: This transformations were living inside the original codebase, are they really necessary?
        # if mean_per_class.ndim == 1:
        #     mean_per_class = mean_per_class.unsqueeze(0)
        # if X.ndim == 1:
        #     X = X.unsqueeze(0)

        pca_concat = []
        for current_row, value_to_infer in enumerate(y):
            class_index = value_to_infer.item() if torch.is_tensor(value_to_infer) else value_to_infer
            assert class_index <= mean_per_class.size(0) - 1, "Is impossible that the index is bigger than the classes size!"
            row = torch.cat((x[current_row], global_mean, mean_per_class[class_index]))
            pca_concat.append(row)
        return torch.vstack(pca_concat)