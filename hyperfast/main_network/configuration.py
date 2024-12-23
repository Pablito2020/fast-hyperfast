from pydantic import BaseModel


class MainNetworkConfig(BaseModel):
    max_categories: int
    number_of_layers: int


DEFAULT_MAIN_NETWORK_CONFIGURATION = MainNetworkConfig(
    number_of_layers=3,
    max_categories=46,
)
