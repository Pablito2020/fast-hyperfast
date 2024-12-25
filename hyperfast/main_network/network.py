from typing import List, Tuple

import torch
from torch import nn, Tensor

from hyperfast.utils.cuda import is_torch_pca


class MainNetwork(nn.Module):
    
    def __init__(self, random_features_net, pca, main_network_weights: List[Tuple[Tensor, Tensor]]):
        super().__init__()
        self.random_features_net = random_features_net
        self.pca_mean = (
            nn.Parameter(pca.mean_)
            if is_torch_pca()
            else nn.Parameter(torch.from_numpy(pca.mean_))
        )
        self.layers = nn.ModuleList()
        for matrix, bias in main_network_weights:
            linear_layer = nn.Linear(matrix.shape[0], matrix.shape[1])
            linear_layer.weight = nn.Parameter(matrix.T)
            linear_layer.bias = nn.Parameter(bias)
            self.layers.append(linear_layer)
            
    def forward(self, X, y=None):
        intermediate_activations = [X]
        X = self.random_features_net(X)
        X = X - self.pca_mean
        X = self.pca_components(X)
        X = torch.clamp(X, -self.clip_data_value, self.clip_data_value)

        for n, layer in enumerate(self.layers):
            if n % 2 == 0:
                residual_connection = X

            X = layer(X)
            if n % 2 == 1 and n < len(self.layers) - 1:
                X = X + residual_connection

            if n < len(self.layers) - 1:
                X = F.relu(X)
                if n == len(self.layers) - 2:
                    intermediate_activations.append(X)
        return X
