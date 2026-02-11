import torch
import torch.nn as nn

# value function network
def Vnet(dim: int, depth: int, width: int) -> nn.Sequential:
    layers = []
    layers.append(nn.Linear(dim, width))
    layers.append(nn.ELU())
    for _ in range(depth-1):
        layers.append(nn.Linear(width, width))
        layers.append(nn.ELU())
    layers.append(nn.Linear(width, 1))
    return nn.Sequential(*layers)

# value function gradient network
def Znet(dim: int, depth: int, width: int) -> nn.Sequential:
    layers = []
    layers.append(nn.Linear(dim, width))
    layers.append(nn.ELU())
    for _ in range(depth-1):
        layers.append(nn.Linear(width, width))
        layers.append(nn.ELU())
    layers.append(nn.Linear(width, dim))
    return nn.Sequential(*layers)