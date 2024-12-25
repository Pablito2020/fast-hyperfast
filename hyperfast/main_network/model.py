from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from hyperfast.main_network.network import MainNetwork
from hyperfast.standardize_data.inference import InferenceStandardizer
from hyperfast.utils.cuda import get_device


@dataclass(frozen=True)
class MainNetworkClassifier:
    standardizer: InferenceStandardizer
    networks: List[MainNetwork]
    classes: np.ndarray
    batch_size: int


    def _predict(self, x) -> np.ndarray:
        X = self.standardizer.preprocess_inference_data(x)
        X_dataset = torch.utils.data.TensorDataset(X)
        X_loader = torch.utils.data.DataLoader(X_dataset, batch_size=self.batch_size, shuffle=False)
        responses = []
        for X_batch in X_loader:
            X_ = X_batch[0].to(get_device())
            with torch.no_grad():
                networks_result = []
                for network in self.networks:
                    # TODO: This is important!
                    # X_transformed = transform_data_for_main_network(
                    #     X=X_, cfg=self._cfg, rf=rf, pca=pca
                    # )
                    x_transformed = X_
                    logit_outputs = network.forward(x_transformed)
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

