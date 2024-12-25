from pydantic import BaseModel, HttpUrl


class HyperNetworkConfig(BaseModel):
    number_of_dimensions: int
    number_of_layers: int
    hidden_size: int

DEFAULT_HYPER_NETWORK_CONFIGURATION = HyperNetworkConfig(
    number_of_dimensions=784,
    number_of_layers=4,
    hidden_size=1024,
)

class LoaderConfig(BaseModel):
    model_path: str = "hyperfast.ckpt"
    model_url: HttpUrl = "https://figshare.com/ndownloader/files/43484094"
