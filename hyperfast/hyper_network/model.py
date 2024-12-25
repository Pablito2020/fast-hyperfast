from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import torch

from hyperfast.data_processing.training import TrainingDataProcessor
from hyperfast.hyper_network.network import HyperNetwork
from hyperfast.main_network.network import MainNetwork
from hyperfast.utils.cuda import get_device


class HyperNetworkGenerator:
    def __init__(self, network: HyperNetwork, processor: TrainingDataProcessor = TrainingDataProcessor(),
                 n_ensemble: int = 16) -> None:
        self.n_ensemble = n_ensemble
        self.processor = processor
        self._model = network
        self.configuration = network.config

    def generate_main_network_from_dataset(self, x: np.ndarray | pd.DataFrame,
                                           y: np.ndarray | pd.Series) -> List[MainNetwork]:
        """
        Generates a main model for the given data.

        Args:
            x (array-like): Input features.
            y (array-like): Target values.
        """
        processed_data = self.processor.sample(x, y)
        _x, _y = processed_data.data
        n_classes = len(processed_data.misc.classes)
        models = []
        device = get_device()
        for n in range(self.n_ensemble):
            _x, _y = _x.to(device), _y.to(device)
            with torch.no_grad():
                models.append(self._model(_x, _y, n_classes))
        return models
