from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class MainNetworkModel:
    models: List[any]
