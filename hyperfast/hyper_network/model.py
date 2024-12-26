from __future__ import annotations

from typing import List, Tuple

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
    def __init__(
        self,
        network: HyperNetwork,
        processor: TrainingDataProcessor = TrainingDataProcessor(),
        n_ensemble: int = 16,
    ) -> None:
        self.n_ensemble = n_ensemble
        self.processor = processor
        self._model = network
        self.configuration = network.config

    def generate_classifier_for_dataset(
        self, x: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series
    ) -> MainNetworkClassifier:
        """
        Generates a Main Network for the given data.

        Args:
            x (array-like): Input features.
            y (array-like): Target values.
        """
        processed_data = self.processor.sample(x, y)
        _x, _y = processed_data.data
        n_classes = len(processed_data.misc.classes)
        networks = []
        device = get_device()
        for _ in tqdm(
            range(self.n_ensemble),
            desc="Generating Main Networks from HyperNetwork... 🧠",
        ):
            _x, _y = _x.to(device), _y.to(device)
            with torch.no_grad():
                network = self._model(_x, _y, n_classes)
                networks.append(network)
        inference_standardizer = InferenceStandardizer.from_training_data(
            numerical_feature_ids=processed_data.misc.numerical_feature_ids,
            categorical_features=processed_data.misc.categorical_features,
            transformers=processed_data.misc.transformers,
        )
        return MainNetworkClassifier(
            networks=networks,
            classes=processed_data.misc.classes,
            standardizer=inference_standardizer,
            batch_size=self.processor.config.batch_size,
        )

    @staticmethod
    def load_from_pre_trained(
        n_ensemble: int = 16,
        model_path="hyperfast.ckpt",
        model_url="https://figshare.com/ndownloader/files/43484094",
    ) -> HyperNetworkGenerator:
        ModelDownloader.download_model(model_url=model_url, model_path=model_path)
        device = get_device()
        network = HyperNetwork(
            config=DEFAULT_HYPER_NETWORK_CONFIGURATION,
            main_network_config=DEFAULT_MAIN_NETWORK_CONFIGURATION,
        )
        print(f"Loading Hyper Network on device: {device}... ⏰", flush=True)
        network.load_state_dict(
            torch.load(model_path, map_location=torch.device(device), weights_only=True)
        )
        network.eval()
        print(f"Loaded Hyper Network on device: {device} successfully! 🚀", flush=True)
        return HyperNetworkGenerator(network=network, n_ensemble=n_ensemble)

    def meta_train(
        self,
        datasets: List[Tuple[np.ndarray | pd.DataFrame, np.ndarray | pd.DataFrame]],
    ):
        print("Beginning meta-training on hyper network....")
        meta_training_datasets = []
        for x, y in datasets:
            processed_data = self.processor.sample(x, y)
            _x, _y = processed_data.data
            n_classes = len(processed_data.misc.classes)
            meta_training_datasets.append((_x, _y, n_classes))
        self._model.meta_train(datasets=meta_training_datasets, epochs=3)
