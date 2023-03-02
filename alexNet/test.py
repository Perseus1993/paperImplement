import numpy as np
import matplotlib.pyplot as plt
import torch
from torchviz import make_dot

from alexNet.network import AlexNet

if __name__ == '__main__':
    x = torch.randn(1, 3, 224, 224).requires_grad_(True)
    network = AlexNet(101)
    y = network(x)

    pic = make_dot(y, params=dict(list(network.named_parameters()) + [('x', x)]))
    pic.format = "png"
    pic.directory = "data"
    pic.view()
