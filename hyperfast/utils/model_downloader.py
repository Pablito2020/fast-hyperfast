import os

import requests
from tqdm import tqdm


class ModelDownloader:
    @staticmethod
    def download_model(model_url: str, model_path: str):
        if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
            print(
                f"The model weights already exists at '{model_path}' and is not empty. Skipping download."
            )
            return
        response = requests.get(model_url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        with (
            open(model_path, "wb") as f,
            tqdm(
                desc=f"Downloading model weights from {model_url}...🌐",
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar,
        ):
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
                bar.update(len(chunk))
