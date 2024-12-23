from typing import Literal

from pydantic import BaseModel, HttpUrl


class HyperNetworkConfig(BaseModel):
    number_of_dimensions: int
    number_of_layers: int
    hidden_size: int


class LoaderConfig(BaseModel):
    load_device: Literal["cpu", "cuda"] = "cpu"
    model_path: str = "hyperfast.ckpt"
    model_url: HttpUrl = "https://figshare.com/ndownloader/files/43484094"
