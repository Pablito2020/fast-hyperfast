import torch
import torch.nn as nn

from hyperfast.hyper_network.configuration import HyperNetworkConfig
from hyperfast.main_network.configuration import MainNetworkConfig


class HyperNetwork(nn.Module):

    def __init__(self, config: HyperNetworkConfig, main_network_config: MainNetworkConfig, loaded_on: str):
        super().__init__()
        self.__configuration = config
        self.loaded_on = loaded_on
        middle_layers = []
        for n in range(config.number_of_layers - 2):
            middle_layers.append(nn.Linear(config.hidden_size, config.hidden_size))
            middle_layers.append(nn.ReLU())
        self.number_input_features = config.number_of_dimensions + main_network_config.max_categories

        self.hypernetworks = nn.ModuleList()
        self.hn_emb_to_weights = nn.ModuleList()

        for n in range(main_network_config.number_of_layers - 1):
            if n > 0:
                self.number_input_features = config.number_of_dimensions * 2 + main_network_config.max_categories
            num_input_features_hn = self.number_input_features + config.number_of_dimensions * 2

            hn_layers = [nn.Linear(num_input_features_hn, config.hidden_size), nn.ReLU()]
            hn_layers = hn_layers + middle_layers

            self.hypernetworks.append(nn.Sequential(*hn_layers))
            output_size_hn = (config.number_of_dimensions + 1) * config.number_of_dimensions
            self.hn_emb_to_weights.append(
                nn.Linear(config.hidden_size, output_size_hn)
            )

        hn_layers = []
        last_hn_output_size = config.number_of_dimensions + 1
        self.number_input_features += config.number_of_dimensions * 2

        hn_layers.append(nn.Linear(self.number_input_features, config.hidden_size))
        hn_layers.append(nn.ReLU())
        hn_layers = hn_layers + middle_layers
        hn_layers.append(nn.Linear(config.hidden_size, last_hn_output_size))
        self.hypernetworks.append(nn.Sequential(*hn_layers))
        self.nn_bias = nn.Parameter(torch.ones(2))

    def forward(self, x, y, n_classes):
        print(n_classes)
        pass

    @property
    def config(self) -> HyperNetworkConfig:
        return self.__configuration

    # def train(self, x, y, n_classes):
    #     raise NotImplementedError("Not implemented yet. It is explained inside the \"hyperfast and baselines implementation\" section on the paper")

