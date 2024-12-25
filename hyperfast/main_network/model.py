from __future__ import annotations

from dataclasses import dataclass
from typing import List

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


@dataclass(frozen=True)
class MainNetworkClassifier:
    standardizer: InferenceStandardizer
    networks: List[MainNetwork]
    classes: np.ndarray
    batch_size: int

    def _predict(self, x) -> np.ndarray:
        preprocessed_x = self.standardizer.preprocess_inference_data(x)
        x_dataset = torch.utils.data.TensorDataset(preprocessed_x)
        x_loader = torch.utils.data.DataLoader(x_dataset, batch_size=self.batch_size, shuffle=False)
        responses = []
        for x_batch in x_loader:
            x_ = x_batch[0].to(get_device())
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
        outputs = self._predict(x)
        return self.classes[np.argmax(outputs, axis=1)]

    def fine_tune_networks(self, x, y, optimize_steps: int, learning_rate: float = 0.0001):
        tune_standardizer = TrainingDataProcessor()
        res = tune_standardizer.sample(x, y)
        pre_processed_x, pre_processed_y = res.data
        for network_index in range(len(self.networks)):
            self.fine_tune_network_index(pre_processed_x, pre_processed_y, optimize_steps, network_index, learning_rate)

    def fine_tune_network_index(self, x, y, optimize_steps: int, index: int, learning_rate: float):
        assert index < len(self.networks), "You can't optimize a network that doesn't exist!"
        dataset = TensorDataset(x, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        network = self.networks[index]
        optimizer = optim.AdamW(network.parameters(), lr=learning_rate)
        device = get_device()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=10
        )
        for step in tqdm(range(optimize_steps), desc=f"Fine Tunning Network {index + 1} 📖"):
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = network(inputs, targets)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                print(f"[Fine Tune (Network {index + 1})] Step: [{step + 1}/{optimize_steps}], Loss: {loss.item()}")

            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(loss.item())
                else:
                    scheduler.step()

    def save_model(self, path: str):
        torch.save(self.networks, f"{path}/main_networks.pth")
        np.save(f'{path}/classes.npy', self.classes)
        print(f"Current batch size is: {self.batch_size}")

    @staticmethod
    def load_from_pre_trained(x, y, path: str, batch_size: int) -> MainNetworkClassifier:
        t = TrainingDataProcessor()
        device = get_device()
        res = t.sample(x, y)
        inference_standardizer = InferenceStandardizer(data=res)
        print(f"Loading Main Model on device: {device}... ⏰", flush=True)
        networks = torch.load(f"{path}/main_networks.pth", map_location=torch.device(device))
        classes = np.load(f'{path}/classes.npy')
        print(f"Loaded Main Model on device: {device} successfully! 🚀", flush=True)
        return MainNetworkClassifier(
            standardizer=inference_standardizer,
            networks=networks,
            classes=classes,
            batch_size=batch_size
        )
