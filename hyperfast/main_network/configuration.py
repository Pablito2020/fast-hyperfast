from pydantic import BaseModel


class MainNetworkConfig(BaseModel):
    max_categories: int
    number_of_layers: int
