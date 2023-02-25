import sys

from torch.utils.benchmark import timer

sys.path.append("/home/urbanbigdata/deploy")
from PIL import Image
import configparser
import os
import multiprocessing as mp
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader
from torchvision.datasets import Food101
from torchvision import transforms, utils

from alexNet.network import AlexNet

LEARNING_RATE = [0.0001, 0.0002, 0.0003, 0.0004]
IS_NORM = [True,False]
OPTIMIZER = ["adam","sdg"]

res = itertools.product(LEARNING_RATE, IS_NORM, OPTIMIZER, repeat=1)
plans = list(res)

filename = 'res.txt'


def data_process(is_norm):
    config = configparser.ConfigParser()
    config.read("../config/conf.ini")

    root_file = os.path.join(config["data"].get("root_file"), "Food101")
    model_save_file = config["data"].get("save_file")

    sample_data = Food101(root=root_file, split="test", transform=transforms.ToTensor(), download=True)

    sample_size = len(sample_data)
    sampler1 = torch.utils.data.sampler.SubsetRandomSampler(
        np.random.choice(range(len(sample_data)), sample_size))
    transform_list_train = [transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor()]
    transform_list_test = [transforms.Resize((224, 224)),
                           transforms.ToTensor()]

    if is_norm:
        transform_list_train.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        transform_list_test.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

    transform_train = transforms.Compose(transform_list_train)
    transform_test = transforms.Compose(transform_list_test)

    train_data = Food101(root=root_file, split="train", transform=transform_train, download=True)
    test_data = Food101(root=root_file, split="test", transform=transform_test, download=True)
    return train_data, test_data, model_save_file


def calculate(path):
    files = os.listdir(path)
    trans = transforms.ToTensor()
    mean = torch.zeros(3)
    std = torch.zeros(3)

    for file in tqdm(files):
        img = Image.open(path + file)
        img = trans(img)
        for i in range(3):  # RGB图像的通道数，因为 transforms.Normalize()是在通道上进行归一化
            mean[i] += img[i, :, :].mean()
            std[i] += img[i, :, :].std()
    mean.div_(files.__len__())
    std.div_(files.__len__())
    print(mean, std)


def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    epochs = 50

    for plan in plans:
        lr = plan[0]
        is_norm = plan[1]
        optim_name = plan[2]
        cur_plan = 'lr-%s norm-%s optim_name-%s' % (lr, is_norm, optim_name)
        print(cur_plan)
        train_data, test_data, model_save_file = data_process(is_norm)
        model_save_file = os.path.join(model_save_file, ('AlexNet' + cur_plan + '.pth'))
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=mp.cpu_count())
        test_loader = DataLoader(test_data, batch_size=4, shuffle=False, num_workers=mp.cpu_count())
        # print("using {} images for training, {} images for validation.".format(len(train_data), len(test_data)))

        food_list = train_data.class_to_idx
        cla_dict = dict((val, key) for key, val in food_list.items())
        test_data_iter = iter(test_loader)
        test_image, test_label = next(test_data_iter)
        # print(test_image)
        # print(test_label)

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
        if optim_name == "adam":
            optimizer = optim.Adam(network.parameters(), lr=lr)
        else:
            optimizer = optim.SGD(network.parameters(), lr=lr)

        train_time_start = timer()
        stat_data = np.zeros((epochs, 2))
        for epoch in range(epochs):
            print(f"Epoch: {epoch}\n-------")
            network.train()
            train_loss = 0.0
            for data in tqdm(train_loader):
                img, labels = data
                optimizer.zero_grad()
                out = network(img.to(device))
                loss = loss_fun(out, labels.to(device))
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)
            # print("train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, train_loss))

            test_loss, test_acc = 0, 0
            with torch.inference_mode():
                for test_img, test_label in test_loader:
                    test_pred = network(test_img.to(device))
                    test_loss += loss_fun(test_pred, test_label.to(device))
                    test_acc += accuracy_fn(y_true=test_label.to(device), y_pred=test_pred.argmax(dim=1))
                test_loss /= len(test_loader)
                test_acc /= len(test_loader)
                stat_data[epoch][0] = train_loss
                stat_data[epoch][1] = test_acc
            print(f"\nTrain loss: {train_loss:.5f} | Test loss: {test_loss:.5f} | Test Acc: {test_acc:.2f}% \n")
        plt.plot(stat_data, label=cur_plan)
        torch.save(network.state_dict(), model_save_file)

        train_time_end = timer()
        print(f"Train time : {train_time_end - train_time_start:.3f} seconds")

        plt.savefig("res.png")
        plt.show()


if __name__ == '__main__':
    train()
