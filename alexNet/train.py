import configparser
import os
import multiprocessing as mp
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader
from torchvision.datasets import Food101
from torchvision import transforms, utils

from alexNet.network import AlexNet


def data_process():
    config = configparser.ConfigParser()
    config.read("../config/conf.ini")
    print(config.sections())
    root_file = os.path.join(config["data"].get("root_file"), "Food101")
    print(root_file)
    transform_train = transforms.Compose([transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor()
                                          ])
    transform_test = transforms.Compose([transforms.Resize((224, 224)),
                                         transforms.ToTensor()])

    train_data = Food101(root=root_file, split="train", transform=transform_train, download=True)
    test_data = Food101(root=root_file, split="test", transform=transform_test, download=True)
    return train_data, test_data


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    train_data, test_data = data_process()
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=mp.cpu_count())
    test_loader = DataLoader(test_data, batch_size=4, shuffle=False, num_workers=mp.cpu_count())
    print("using {} images for training, {} images for validation.".format(len(train_data), len(test_data)))

    food_list = train_data.class_to_idx
    cla_dict = dict((val, key) for key, val in food_list.items())
    test_data_iter = iter(test_loader)
    test_image, test_label = next(test_data_iter)
    print(test_image)
    print(test_label)

    # def imshow(img):
    #     npimg = img.numpy()
    #     plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #     plt.show()
    #
    # print(' '.join('%5s' % cla_dict[test_label[j].item()] for j in range(4)))
    # imshow(utils.make_grid(test_image))
    network = AlexNet(101)
    network.to(device)
    loss_fun = nn.CrossEntropyLoss()
    optimizer = optim.Adam(network.parameters(), lr=0.0002)
    epochs = 10
    save_path = './AlexNet.pth'
    best_acc =0.0
    train_steps = len(train_loader)
    for epoch in tqdm(range(epochs)):
        print(f"Epoch: {epoch}\n-------")
        network.train()
        train_loss = 0.0
        for batch,data in enumerate(train_loader):
            img, labels = data
            optimizer.zero_grad()
            out = network(img.to(device))
            loss = loss_fun(out,labels.to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        print("train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,epochs,train_loss))

        test_loss = 0
        with torch.inference_mode():
            for test_img, test_label in test_loader:
                test_pred = network(test_img)
                test_loss += loss_fun(test_pred, test_label)
            test_loss /= len(test_loader)
        print(f"\nTrain loss: {train_loss:.5f} | Test loss: {test_loss:.5f} \n")


if __name__ == '__main__':
    train()
