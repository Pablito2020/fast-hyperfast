from hyperfast.hyper_network.loader import HyperNetworkLoader
from hyperfast.main_network.configuration import MainNetworkConfig
from hyperfast.hyper_network.configuration import HyperNetworkConfig

hyper_network_config = HyperNetworkConfig(
    number_of_dimensions=784,
    number_of_layers=4,
    hidden_size=1024,
)

main_network_config = MainNetworkConfig(
    number_of_layers=3,
    max_categories=46,
)

net = HyperNetworkLoader.get_loaded_network(config=hyper_network_config, main_network_config=main_network_config)
for p in net.parameters():
    print(p)
net.forward(None, None, None)