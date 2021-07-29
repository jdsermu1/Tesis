##
import random
import sys
from collections import OrderedDict
import math
import torch
import os
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Lambda, Compose, Resize, RandomCrop, Normalize, RandomHorizontalFlip, \
    RandomVerticalFlip, RandomRotation, CenterCrop, InterpolationMode
from torch import nn, optim
from torchvision import models
from skimage import io
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import torchvision
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

##



preprocessing = "original"
input_size = 540
strategy = "strategy4"
backbone_name = "Li2019-1"
lr = 1e-3
optimizer_name = "Adam"

model_name = preprocessing + str(input_size) + strategy + backbone_name + optimizer_name
##
random_seed = 5

init_epoch = 0

best_loss = sys.float_info.max

epochs = 65

batch_size = 35

num_classes = 5

device = "cuda" if torch.cuda.is_available() else "cpu"

database_folder = os.path.join("..", "Database")

images_folder = os.path.join(database_folder, "preprocessing images", preprocessing)

annotations_file = os.path.join(database_folder, "labels",
                                f"labelsPreprocessing{preprocessing.capitalize()}.csv")

run = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

writer = SummaryWriter(os.path.join(database_folder, "runs", run))

model_path = os.path.join(database_folder, "models", model_name+".pt")

##


def strategy1(labels):
    size_validation = 100
    size_test = 312
    size_partition = size_test + size_validation
    labels_strategy = pd.DataFrame({})
    for c in range(5):
        labels_c = labels[labels['level'] == c]
        labels_train, labels_test_val = train_test_split(labels_c, test_size=size_partition, random_state=random_seed)
        labels_train["set"] = "train"
        labels_test, labels_validation = train_test_split(labels_test_val, test_size=size_validation,
                                                          random_state=random_seed)
        labels_test["set"] = "test"
        labels_validation["set"] = "validation"
        labels_strategy = pd.concat([labels_strategy, labels_validation, labels_test, labels_train])
    return labels_strategy


def strategy2(labels):
    size_validation = 100
    size_test = 312
    size_train = 1502
    size_partition = size_test + size_validation
    labels_strategy = pd.DataFrame({})
    for c in range(5):
        labels_c = labels[labels['level'] == c]
        labels_train, labels_test_val = train_test_split(labels_c, test_size=size_partition, random_state=random_seed)
        if len(labels_train) > size_train:
            labels_train, _ = train_test_split(labels_train, train_size=size_train, random_state=random_seed)
        labels_train["set"] = "train"
        labels_test, labels_validation = train_test_split(labels_test_val, test_size=size_validation,
                                                          random_state=random_seed)
        labels_test["set"] = "test"
        labels_validation["set"] = "validation"
        labels_strategy = pd.concat([labels_strategy, labels_validation, labels_test, labels_train])
    return labels_strategy


def strategy3(labels):
    size_validation = 100
    size_test = 312
    size_train = 15000
    size_partition = size_validation + size_test
    labels_train = pd.DataFrame({})
    labels_test = pd.DataFrame({})
    labels_validation = pd.DataFrame({})
    for c in range(5):
        labels_c = labels[labels['level'] == c]
        labels_train_c, labels_test_val_c = train_test_split(labels_c, test_size=size_partition,
                                                             random_state=random_seed)
        labels_validation_c, labels_test_c = train_test_split(labels_test_val_c, test_size=size_test,
                                                              random_state=random_seed)
        labels_train = pd.concat([labels_train, labels_train_c])
        labels_test = pd.concat([labels_test, labels_test_c])
        labels_validation = pd.concat([labels_validation, labels_validation_c])


    rus = RandomUnderSampler(sampling_strategy={0: size_train}, random_state=random_seed)
    ros = RandomOverSampler(sampling_strategy="not majority", random_state=random_seed)
    labels_train, _ = rus.fit_sample(labels_train, labels_train["level"])
    labels_train, _ = ros.fit_sample(labels_train, labels_train["level"])
    labels_train["set"] = "train"
    labels_test["set"] = "test"
    labels_validation["set"] = "validation"
    labels_strategy = pd.concat([labels_train, labels_validation, labels_test])
    return labels_strategy


def strategy4(labels):
    test_size = 15600
    validation_size = 5000
    train_size = 15000
    labels_train, labels_test_validation = train_test_split(labels, test_size=test_size+validation_size,
                                                            stratify=labels["level"], random_state=random_seed)
    labels_validation, labels_test = train_test_split(labels_test_validation, test_size=test_size,
                                                      stratify=labels_test_validation["level"],
                                                      random_state=random_seed)
    rus = RandomUnderSampler(sampling_strategy={0: train_size}, random_state=random_seed)
    ros = RandomOverSampler(sampling_strategy="not majority", random_state=random_seed)
    labels_train, _ = rus.fit_sample(labels_train, labels_train["level"])
    labels_train, _ = ros.fit_sample(labels_train, labels_train["level"])
    labels_train["set"] = "train"
    labels_validation["set"] = "validation"
    labels_test["set"] = "test"
    labels_strategy = pd.concat([labels_train, labels_validation, labels_test])
    return labels_strategy


if strategy == "strategy1":
    func = strategy1
elif strategy == "strategy2":
    func = strategy2
elif strategy == "strategy3":
    func = strategy3
elif strategy == "strategy4":
    func = strategy4
else:
    func = strategy1
labels_df = pd.read_csv(annotations_file)
labels_df = func(labels_df)

##


class CustomDataset(Dataset):
    def __init__(self, labels, img_dir, folder="train", transform=None, target_transform=None):
        self.img_dir = img_dir
        self.img_labels = labels
        self.img_labels = self.img_labels[self.img_labels["set"] == folder]
        self.transform = transform
        self.target_transform = target_transform
        self.folder = folder

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        label = self.img_labels.iloc[idx, 1]
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0] + ".jpeg")
        image = io.imread(img_path)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def build_data_loaders():
    array_train = [ToTensor()]
    array = [ToTensor()]
    if preprocessing == "original":
        array_train.append(CenterCrop((540, 540)))
        array.append((CenterCrop((540, 540))))
    if input_size != 540:
        array_train.append(Resize(input_size))
        array.append(Resize(input_size))
    array_train.append(RandomVerticalFlip(0.5))
    array_train.append(RandomHorizontalFlip(0.5))
    array_train.append(RandomRotation((0, 360)))

    preprocess_train = Compose(array_train)

    preprocess = Compose(array)

    train_dataset = CustomDataset(labels_df, images_folder, transform=preprocess_train)
    validation_dataset = CustomDataset(labels_df, images_folder, folder="validation", transform=preprocess)
    test_dataset = CustomDataset(labels_df, images_folder, folder="test", transform=preprocess)

    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True), \
           DataLoader(validation_dataset, batch_size=batch_size, shuffle=True), \
           DataLoader(test_dataset, batch_size=batch_size, shuffle=True)



##
class BaseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, fractional, dropout, padding="same",
                 alpha=0.333, output_ratio=1.0/1.8):
        super(BaseBlock, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.leakyRelu = nn.LeakyReLU(negative_slope=alpha)
        self.fractional = fractional
        self.p = dropout
        if self.fractional:
            self.fractionalMaxPooling = nn.FractionalMaxPool2d(kernel_size, output_ratio=output_ratio)
        if self.p > 0.0:
            self.dropout = nn.Dropout2d(p=dropout)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.leakyRelu(x)
        if self.fractional:
            x = self.fractionalMaxPooling(x)
        if self.p > 0.0:
            x = self.dropout(x)
        return x




class ModelLi2019(nn.Module):
    def __init__(self, blocks):
        super(ModelLi2019, self).__init__()
        array = []
        for i, block in enumerate(blocks):
            in_channels = 3 if i == 0 else blocks[i-1][0]
            array.append((f"baseBlock{i}", BaseBlock(in_channels, block[0], block[1],  block[2], block[3])))
        self.features = nn.Sequential(OrderedDict(array))
        self.linear = nn.Linear(1424, 5)
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        x = self.logSoftmax(x)
        return x


def modelLi2019_1():
    blocks = [(32, 5, True, -1), (64, 3, True, -1), (96, 3, True, -1), (128, 3, True, -1),  (160, 3, True, -1),
              (192, 3, True, -1), (224, 3, True, -1), (256, 3, True, 32.0/352), (288, 2, True, 32.0/384),
              (320, 3, False, 64.0/416), (356, 1, False, 64.0/448)]
    return ModelLi2019(blocks)


def construct_model():

    backbone = modelLi2019_1()

    backbone = backbone.to(device)

    if optimizer_name == "Adam":
        opt = optim.Adam(backbone.parameters(), lr=lr)
    elif optimizer_name == "SGD":
        opt = optim.SGD(backbone.parameters(), lr=lr)
    elif optimizer_name == "RMSprop":
        opt = optim.RMSprop(backbone.parameters(), lr=lr)
    else:
        opt = optim.SGD(backbone.parameters(), lr=lr)

    return backbone, opt

##


train_dataloader, validation_dataloader, test_dataloader = build_data_loaders()
model, optimizer = construct_model()
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda ep: math.exp(-ep*0.05))
criterion = nn.NLLLoss()

##

if os.path.exists(model_path):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    init_epoch = checkpoint['epoch'] + 1
    best_loss = checkpoint['best_loss']


##

images, labels = iter(train_dataloader).next()
img_grid = torchvision.utils.make_grid(images)
writer.add_image('train_example', img_grid)


##




def train(epoch):
    model.train()
    size = len(train_dataloader.dataset)
    number_batches = len(train_dataloader)
    running_loss, correct = 0.0, 0
    running_predictions, running_labels = np.array([]), np.array([])
    for i, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        running_labels = np.concatenate([running_labels, y.to("cpu").numpy()])
        running_predictions = np.concatenate([running_predictions, pred.argmax(1).to("cpu").numpy()])
        if i % (int(number_batches / 10)) == 0 and i != 0:
            print(f"loss: {running_loss / (int(number_batches / 10)):>7f}  [{i * len(X):>5d}/{size:>5d}]")
            writer.add_scalar('Loss/Training',
                              running_loss / (int(number_batches / 10)),
                              epoch * len(train_dataloader) + i)
            writer.add_scalar('Accuracy/Training',
                              correct / len(running_predictions),
                              epoch * len(train_dataloader) + i)

            precision, recall, f1_score, _ = precision_recall_fscore_support(running_labels, running_predictions,
                                                                             labels=[0, 1, 2, 3, 4])
            for j in range(5):
                writer.add_scalar(f'Precision/Training/Class_{str(j)}',
                                  precision[j],
                                  epoch * len(train_dataloader) + i)
                writer.add_scalar(f'Recall/Training/Class_{str(j)}',
                                  recall[j],
                                  epoch * len(train_dataloader) + i)
                writer.add_scalar(f'F1_Score/Training/Class_{str(j)}',
                                  f1_score[j],
                                  epoch * len(train_dataloader) + i)
            running_loss, correct = 0.0, 0
            running_predictions, running_labels = np.array([]), np.array([])


def validation(epoch):
    global best_loss
    model.eval()
    size = len(validation_dataloader.dataset)
    num_batches = len(validation_dataloader)
    test_loss, correct = 0, 0
    all_predictions = np.array([])
    all_labels = np.array([])
    with torch.no_grad():
        for X, y in validation_dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += criterion(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            all_labels = np.concatenate([all_labels, y.to("cpu").numpy()])
            all_predictions = np.concatenate([all_predictions, pred.argmax(1).to("cpu").numpy()])
    test_loss /= num_batches
    correct /= size
    print(f"Validation Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    writer.add_scalar('Loss/Validation',
                      test_loss,
                      epoch + 1)
    writer.add_scalar('Accuracy/Validation',
                      correct * 100,
                      epoch + 1)
    precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels,
                                                                     all_predictions, labels=[0, 1, 2, 3, 4])
    for i in range(5):
        writer.add_scalar(f'Precision/Validation/Class_{str(i)}',
                          precision[i],
                          epoch)
        writer.add_scalar(f'Recall/Validation/Class_{str(i)}',
                          recall[i],
                          epoch)
        writer.add_scalar(f'F1_Score/Validation/Class_{str(i)}',
                          f1_score[i],
                          epoch)

    if test_loss < best_loss:
        best_loss = test_loss
        dict_save = {'epoch': epoch, 'model_state_dict': model.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict(), 'best_loss': best_loss,
                     'scheduler_state_dict': scheduler.state_dict()}
        torch.save(dict_save, model_path)




for t in range(init_epoch, epochs):
    print(f"Epoch {t}\n-------------------------------")
    train(t)
    validation(t)
    scheduler.step()
print("Done!")

writer.add_hparams({
    "Preprocessing": preprocessing,
    "Data augmentation strategy": strategy,
    "Backbone": backbone_name,
    "Learning rate": lr,
    "Optimizer": optimizer_name,
    "Input Size": input_size,
}, {
    "Best Loss": best_loss
})


writer.flush()
writer.close()
