import torch.nn as nn


class Generator(nn.Module):
    def __init__(
            self,
            depth=10,
            num_channels=3,
            latent_size=512,
    ):
        super().__init__()
        self.depth = depth
        self.num_channels = num_channels

