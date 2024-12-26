import random
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

from hyperfast.hyper_network.configuration import (
    HyperNetworkConfig,
    DEFAULT_CLIP_DATA_VALUE,
)
from hyperfast.hyper_network.embedding import (
    RandomFeatures,
    get_pca,
    get_mean_per_class,
)
from hyperfast.main_network.configuration import MainNetworkConfig
from hyperfast.main_network.network import MainNetwork
from hyperfast.utils.cuda import get_device, is_cuda


class HyperNetwork(nn.Module):
    def __init__(
        self, config: HyperNetworkConfig, main_network_config: MainNetworkConfig
    ):
        super().__init__()
        self.__configuration = config
        self.__main_network_config = main_network_config
        middle_layers = []
        for n in range(config.number_of_layers - 2):
            middle_layers.append(nn.Linear(config.hidden_size, config.hidden_size))
            middle_layers.append(nn.ReLU())
        self.number_input_features = (
            config.number_of_dimensions + main_network_config.max_categories
        )

        self.hypernetworks = nn.ModuleList()
        self.hn_emb_to_weights = nn.ModuleList()

        for n in range(main_network_config.number_of_layers - 1):
            if n > 0:
                self.number_input_features = (
                    config.number_of_dimensions * 2 + main_network_config.max_categories
                )
            num_input_features_hn = (
                self.number_input_features + config.number_of_dimensions * 2
            )

            hn_layers = [
                nn.Linear(num_input_features_hn, config.hidden_size),
                nn.ReLU(),
            ]
            hn_layers = hn_layers + middle_layers

            self.hypernetworks.append(nn.Sequential(*hn_layers))
            output_size_hn = (
                config.number_of_dimensions + 1
            ) * config.number_of_dimensions
            self.hn_emb_to_weights.append(nn.Linear(config.hidden_size, output_size_hn))

        hn_layers = []
        last_hn_output_size = config.number_of_dimensions + 1
        self.number_input_features += config.number_of_dimensions * 2

        hn_layers.append(nn.Linear(self.number_input_features, config.hidden_size))
        hn_layers.append(nn.ReLU())
        hn_layers = hn_layers + middle_layers
        hn_layers.append(nn.Linear(config.hidden_size, last_hn_output_size))
        self.hypernetworks.append(nn.Sequential(*hn_layers))
        self.nn_bias = nn.Parameter(torch.ones(2))

    def forward(self, x, y, n_classes: int) -> MainNetwork:
        x = x.flatten(start_dim=1)

        random_features = RandomFeatures(input_shape=x.shape[1])
        if is_cuda():
            random_features = random_features.cuda()
        x = random_features(x)

        x, pca = get_pca(x, number_dimensions=self.__configuration.number_of_dimensions)
        out = torch.clamp(x, -DEFAULT_CLIP_DATA_VALUE, DEFAULT_CLIP_DATA_VALUE)

        pca_output = get_mean_per_class(x=out, y=y, n_classes=n_classes)
        y_onehot = F.one_hot(y, self.__main_network_config.max_categories)

        # TODO: Clean this
        data = torch.cat((pca_output, y_onehot), dim=1)
        main_network = []
        for n in range(self.__main_network_config.number_of_layers - 1):
            if n % 2 == 0:
                residual_connection = out
            weights = self.__get_main_weights(data=data, n=n)
            out, main_linear_layer = self.__forward_linear_layer(
                out, weights, self.__configuration.number_of_dimensions
            )
            if n % 2 == 0:
                out = F.relu(out)
            else:
                out = out + residual_connection
                out = F.relu(out)
            main_network.append(main_linear_layer)
            data = torch.cat((out, pca_output, y_onehot), dim=1)

        # Last network layer
        data = torch.cat((out, pca_output, y_onehot), dim=1)
        weights_per_sample = self.__get_main_net_last_weights(data)
        weights = []
        last_input_mean = []
        for lab in range(n_classes):
            if torch.sum((y == lab)) > 0:
                w = torch.mean(weights_per_sample[y == lab], dim=0, keepdim=True)
                input_mean = torch.mean(out[y == lab], dim=0, keepdim=True)
            else:
                w = torch.mean(weights_per_sample, dim=0, keepdim=True)
                input_mean = torch.mean(out, dim=0, keepdim=True)
            weights.append(w)
            last_input_mean.append(input_mean)
        weights = torch.cat(weights)
        last_input_mean = torch.cat(last_input_mean)
        weights[:, :-1] = weights[:, :-1] + last_input_mean
        weights = weights.T
        out, last_linear_layer = self.__forward_linear_layer(out, weights, n_classes)
        main_network.append(last_linear_layer)

        return MainNetwork(
            random_features_net=random_features,
            pca=pca,
            main_network_weights=main_network,
        )

    def __get_main_weights(self, data, n: int) -> Tensor:
        """
        Get the weights for the N layer of the main network
        """
        hyper_network = self.hypernetworks[n].to(get_device())
        converter_to_weights = self.hn_emb_to_weights[n].to(get_device())
        emb = hyper_network(data)
        global_emb = torch.mean(emb, dim=0)
        return converter_to_weights(global_emb)

    def __get_main_net_last_weights(self, data) -> Tensor:
        """
        Get the weights for the last layer of the main network
        """
        last_hyper_network = self.hypernetworks[-1].to(get_device())
        return last_hyper_network(data)

    def __forward_linear_layer(
        self, x, weights, number_of_dimensions
    ) -> Tuple[any, Tuple[Tensor, Tensor]]:
        """
        Remember the output of a forward is:
            f(x) = (x * m) + bias
        """
        weights = weights.view(-1, number_of_dimensions)
        m = weights[:-1, :]
        bias = weights[-1, :]
        x = torch.mm(x, m) + bias
        return x, (m, bias)

    def meta_train(
        self,
        datasets: List[Tuple[Tensor, Tensor, int]],
        epochs: int = 64,
        accumulation_steps: int = 25,
    ):
        """
        Meta-Training as explained on the original article (page 12, section HyperFast and Baselines implementation):
        https://arxiv.org/pdf/2402.14335
        """
        self.train()
        device = get_device()
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer.zero_grad()
        step = 0
        for i in tqdm(
            range(epochs * accumulation_steps), desc="Training the HyperNetwork 💊"
        ):
            (x_task, y_task, n_classes) = random.choice(datasets)
            x_task, y_task = x_task.to(device), y_task.to(device)
            main_network = self.forward(x_task, y_task, n_classes)
            main_network.to(device)
            predictions = main_network(x_task)
            loss = criterion(predictions, y_task) / accumulation_steps
            loss.backward()
            if (step + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

    @property
    def config(self) -> HyperNetworkConfig:
        return self.__configuration
