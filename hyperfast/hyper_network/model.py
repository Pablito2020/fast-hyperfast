from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from hyperfast.hyper_network.configuration import DEFAULT_HYPER_NETWORK_CONFIGURATION
from hyperfast.hyper_network.network import HyperNetwork
from hyperfast.main_network.configuration import DEFAULT_MAIN_NETWORK_CONFIGURATION
from hyperfast.main_network.model import MainNetworkClassifier
from hyperfast.standardize_data.inference import InferenceStandardizer
from hyperfast.standardize_data.training import TrainingDataProcessor
from hyperfast.utils.cuda import get_device
from hyperfast.utils.model_downloader import ModelDownloader


class HyperNetworkGenerator:
    def __init__(self, network: HyperNetwork, processor: TrainingDataProcessor = TrainingDataProcessor(),
                 n_ensemble: int = 16) -> None:
        self.n_ensemble = n_ensemble
        self.processor = processor
        self._model = network
        self.configuration = network.config

    def generate_classifier_for_dataset(self, x: np.ndarray | pd.DataFrame,
                                        y: np.ndarray | pd.Series) -> MainNetworkClassifier:
        """
        Generates a main model for the given data.

        Args:
            x (array-like): Input features.
            y (array-like): Target values.
        """
        processed_data = self.processor.sample(x, y)
        _x, _y = processed_data.data
        n_classes = len(processed_data.misc.classes)
        networks = []
        device = get_device()
        for _ in tqdm(range(self.n_ensemble), desc="Generating Main Networks from HyperNetwork... 🧠"):
            _x, _y = _x.to(device), _y.to(device)
            with torch.no_grad():  # Important! Since we're not testing, we're creating the "final" weights
                network = self._model(_x, _y, n_classes)
                networks.append(network)
        return MainNetworkClassifier(networks=networks, classes=processed_data.misc.classes,
            standardizer=InferenceStandardizer(data=processed_data), batch_size=self.processor.config.batch_size)

    @staticmethod
    def load_from_pre_trained(
            n_ensemble: int = 16,
            model_path="hyperfast.ckpt",
            model_url="https://figshare.com/ndownloader/files/43484094"
    ) -> HyperNetworkGenerator:
        ModelDownloader.download_model(model_url=model_url, model_path=model_path)
        device = get_device()
        network = HyperNetwork(config=DEFAULT_HYPER_NETWORK_CONFIGURATION,
                               main_network_config=DEFAULT_MAIN_NETWORK_CONFIGURATION)
        print(f"Loading Hyper Network on device: {device}... ⏰", flush=True)
        network.load_state_dict(
            torch.load(model_path, map_location=torch.device(device), weights_only=True))
        network.eval()
        print(f"Loaded Hyper Network on device: {device} successfully! 🚀", flush=True)
        return HyperNetworkGenerator(
            network=network,
            n_ensemble=n_ensemble
        )
