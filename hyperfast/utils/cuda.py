import torch


def is_cuda() -> bool:
    return torch.cuda.is_available()


def get_device() -> str:
    return "cuda:0" if is_cuda() else "cpu"


def is_torch_pca() -> bool:
    return True
