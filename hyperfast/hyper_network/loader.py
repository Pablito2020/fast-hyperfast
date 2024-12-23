import os
from dataclasses import dataclass, field

import torch

from hyperfast.hyper_network.configuration import HyperNetworkConfig, LoaderConfig, DEFAULT_HYPER_NETWORK_CONFIGURATION
from hyperfast.hyper_network.network import HyperNetwork
from hyperfast.main_network.configuration import MainNetworkConfig, DEFAULT_MAIN_NETWORK_CONFIGURATION


@dataclass(frozen=True)
class HyperNetworkLoader:
    configuration: LoaderConfig = field(default_factory=LoaderConfig)

    def __post_init__(self):
        if not os.path.exists(self.configuration.model_path):
            raise NotImplementedError("TODO: Download the model and show % bar")

    def load_network_from(self, config: HyperNetworkConfig, main_network_config: MainNetworkConfig) -> HyperNetwork:
        network = HyperNetwork(config=config, main_network_config=main_network_config,
                               loaded_on=self.configuration.load_device)
        return self.__load_model_weights(network)

    @staticmethod
    def get_loaded_network(config: HyperNetworkConfig = DEFAULT_HYPER_NETWORK_CONFIGURATION,
                           main_network_config: MainNetworkConfig = DEFAULT_MAIN_NETWORK_CONFIGURATION):
        loader = HyperNetworkLoader()
        return loader.load_network_from(config, main_network_config)

    def __load_model_weights(self, model: HyperNetwork) -> HyperNetwork:
        print(f"Loading model from {self.configuration.model_path} on {self.configuration.load_device} device....",
              flush=True)
        model.load_state_dict(
            torch.load(self.configuration.model_path, map_location=torch.device(self.configuration.load_device),
                       weights_only=True))
        model.eval()
        print(f"Loaded model from {self.configuration.model_path} on {self.configuration.load_device} device!",
              flush=True)
        return model
