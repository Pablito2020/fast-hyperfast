from pydantic import BaseModel


class HyperNetworkConfig(BaseModel):
    number_of_dimensions: int
    number_of_layers: int
    hidden_size: int


DEFAULT_HYPER_NETWORK_CONFIGURATION = HyperNetworkConfig(
    number_of_dimensions=784,
    number_of_layers=4,
    hidden_size=1024,
)

DEFAULT_CLIP_DATA_VALUE = 27.6041
DEFAULT_RANDOM_FEATURE_SIZE = 2**15
