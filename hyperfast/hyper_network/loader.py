import os
from dataclasses import dataclass, field

import torch
from hyperfast.hyper_network.configuration import HyperNetworkConfig, LoaderConfig
from hyperfast.hyper_network.network import HyperNetwork
from hyperfast.main_network.configuration import MainNetworkConfig


@dataclass(frozen=True)
class HyperNetworkLoader:
    configuration: LoaderConfig = field(default_factory=LoaderConfig)

    def __post_init__(self):
        if not os.path.exists(self.configuration.model_path):
            raise NotImplementedError("TODO: Download the model and show % bar")

    def get_loaded_net(self, config: HyperNetworkConfig, main_network_config: MainNetworkConfig) -> HyperNetwork:
        network = HyperNetwork(config=config, main_network_config=main_network_config)
        return self.__load_model_weights(network)

    @staticmethod
    def get_loaded_network(config: HyperNetworkConfig, main_network_config: MainNetworkConfig):
        loader = HyperNetworkLoader()
        return loader.get_loaded_net(config, main_network_config)

    def __load_model_weights(self, model: HyperNetwork) -> HyperNetwork:
        print(f"Loading model from {self.configuration.model_path} on {self.configuration.load_device} device....", flush=True)
        model.load_state_dict(
            torch.load(self.configuration.model_path, map_location=torch.device(self.configuration.load_device), weights_only=True)
        )
        model.eval()
        print(f"Loaded model from {self.configuration.model_path} on {self.configuration.load_device} device!", flush=True)
        return model

