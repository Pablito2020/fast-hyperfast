from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import torch

from hyperfast.main_network.model import MainNetworkClassifier
from hyperfast.standardize_data.training import TrainingDataProcessor
from hyperfast.hyper_network.network import HyperNetwork
from hyperfast.utils.cuda import get_device


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
        for n in range(self.n_ensemble):
            _x, _y = _x.to(device), _y.to(device)
            with torch.no_grad():
                networks.append(self._model(_x, _y, n_classes))
        return MainNetworkClassifier(
            networks=networks,
            classes=processed_data.misc.classes,
            standardizer=None
        )
