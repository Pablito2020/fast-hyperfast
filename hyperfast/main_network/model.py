from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import List, Literal

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from hyperfast.main_network.network import MainNetwork
from hyperfast.standardize_data.inference import InferenceStandardizer
from hyperfast.standardize_data.training import TrainingDataProcessor
from hyperfast.utils.cuda import get_device


@dataclass
class MainNetworkClassifier:
    classes: np.ndarray
    batch_size: int
    standardizer: InferenceStandardizer
    networks: List[MainNetwork]
    device: str = field(default_factory=get_device)

    def _predict(self, x) -> np.ndarray:
        x_dataset = torch.utils.data.TensorDataset(x)
        x_loader = torch.utils.data.DataLoader(
            x_dataset, batch_size=self.batch_size, shuffle=False
        )
        responses = []
        for x_batch in x_loader:
            x_ = x_batch[0].to(self.device)
            with torch.no_grad():
                networks_result = []
                for network in self.networks:
                    logit_outputs = network.forward(x_)
                    predicted = F.softmax(logit_outputs, dim=1)
                    networks_result.append(predicted)
                networks_result = torch.stack(networks_result)
                networks_result = torch.mean(networks_result, axis=0)
                networks_result = networks_result.cpu().numpy()
                responses.append(networks_result)
        return np.concatenate(responses, axis=0)

    def predict(self, x) -> np.ndarray:
        pre_processed_x = self.standardizer.preprocess_inference_data(x)
        outputs = self._predict(pre_processed_x)
        return self.classes[np.argmax(outputs, axis=1)]

    def fine_tune_networks(
        self, x, y, optimize_steps: int, learning_rate: float = 0.0001
    ):
        tune_standardizer = TrainingDataProcessor()
        res = tune_standardizer.sample(x, y)
        pre_processed_x, pre_processed_y = res.data
        for network_index in range(len(self.networks)):
            self.fine_tune_network_index(
                pre_processed_x,
                pre_processed_y,
                optimize_steps,
                network_index,
                learning_rate,
            )

    def fine_tune_network_index(
        self, x, y, optimize_steps: int, index: int, learning_rate: float
    ):
        assert index < len(
            self.networks
        ), "You can't optimize a network that doesn't exist!"
        dataset = TensorDataset(x, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        network = self.networks[index]
        optimizer = optim.AdamW(network.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=10
        )
        for step in tqdm(
            range(optimize_steps), desc=f"Fine Tunning Network {index + 1} 📖"
        ):
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = network(inputs, targets)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                print(
                    f"[Fine Tune (Network {index + 1})] Step: [{step + 1}/{optimize_steps}], Loss: {loss.item()}"
                )
            scheduler.step(metrics=loss.item())

    def save_model(self, path: str):
        # NOTE: This could be improved with python3.13.1, since the copy module has the "replace" function
        # This will allows us to re-enable frozen=True on the dataclass
        # https://docs.python.org/3/library/copy.html#copy.replace
        classifier = copy.deepcopy(self)
        new_networks = [
            net.cpu() if self.device == "cpu" else net.cuda() for net in classifier.networks
        ]
        classifier.networks = new_networks
        joblib.dump(classifier, path)

    @staticmethod
    def load_from_pre_trained(path: str) -> MainNetworkClassifier:
        classifier = joblib.load(path)
        print(f"Loaded classifier. The model is for device: {classifier.device}")
        return classifier
