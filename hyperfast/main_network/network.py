from typing import List, Tuple
import torch.nn.functional as F

import torch
from sklearn.decomposition import PCA
from torch import nn, Tensor

from hyperfast.hyper_network.configuration import DEFAULT_CLIP_DATA_VALUE
from hyperfast.hyper_network.embedding import RandomFeatures, TorchPCA
from hyperfast.utils.cuda import is_torch_pca


class MainNetwork(nn.Module):
    def __init__(
        self,
        random_features_net: RandomFeatures,
        pca: TorchPCA | PCA,
        main_network_weights: List[Tuple[Tensor, Tensor]],
    ):
        super().__init__()
        self.random_features_net = random_features_net
        self.pca_mean = (
            nn.Parameter(pca.mean_)
            if is_torch_pca()
            else nn.Parameter(torch.from_numpy(pca.mean_))
        )
        self.input_features, self.output_features = pca.components_.shape
        self.pca_components = nn.Linear(
            self.input_features, self.output_features, bias=False
        )
        self.pca_components.weight = (
            nn.Parameter(pca.components_)
            if is_torch_pca()
            else nn.Parameter(torch.from_numpy(pca.components_))
        )

        self.layers = nn.ModuleList()
        for matrix, bias in main_network_weights:
            linear_layer = nn.Linear(matrix.shape[0], matrix.shape[1])
            linear_layer.weight = nn.Parameter(matrix.T)
            linear_layer.bias = nn.Parameter(bias)
            self.layers.append(linear_layer)

    def forward(self, x, y=None) -> Tensor:
        """
        Return the logits for the main network
        """
        intermediate_activations = [x]
        x = self.random_features_net(x)
        x = x - self.pca_mean
        x = self.pca_components(x)
        x = torch.clamp(x, -DEFAULT_CLIP_DATA_VALUE, DEFAULT_CLIP_DATA_VALUE)
        for n, layer in enumerate(self.layers):
            if n % 2 == 0:
                residual_connection = x
            x = layer(x)
            if n % 2 == 1 and n < len(self.layers) - 1:
                x = x + residual_connection
            if n < len(self.layers) - 1:
                x = F.relu(x)
                if n == len(self.layers) - 2:
                    intermediate_activations.append(x)
        return x
