import torch.nn as nn
import torch


class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            # in 3*224*224  out 96*55*55 size = (224 - 11 + 补0)/4  + 1 = 55 补0 = 4
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            # in 96*55*55 size = (55 - 3 )/2 + 1  out = 96 * 27 * 27
            nn.MaxPool2d(kernel_size=3, stride=2),
            # in 96 * 27 * 27 size = (27 - 5 + 补0)/1 + 1 = 27     out 256 * 27 * 27
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            # in 256 * 27 * 27 size = (27 - 3 )/2 + 1 = 13  out =  256 * 13 * 13
            nn.MaxPool2d(kernel_size=3, stride=2),
            # in 256 * 13 * 13 size = (13 - 3 + 补0)/1 + 1 = 13 out = 384 * 13 * 13
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # in 384 * 13 * 13 size = (13 - 3 + 补0)/1 + 1  = 13 ,out = 384 * 13 * 13
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # in 384 * 13 * 13 size = (13 - 3 + 补0)/1 + 1 = 13 out = 256 * 13 * 13
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # in =  256 * 13 * 13, out =256 * 6 * 6
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        # batch不展平
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x




