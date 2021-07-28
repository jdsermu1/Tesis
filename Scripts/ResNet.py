##
import random
import sys
from collections import OrderedDict
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


normalize = False
preprocessing = "adaptationDenosing"
input_size = 540
strategy = "strategy4"
backbone_name = "resnet34"
freeze = False
lr = 1e-3
optimizer_name = "Adam"
with_scheduler = True
dense = [1024, 512, 256]



model_name = preprocessing + str(input_size) + strategy + backbone_name + str(freeze) + str(lr) + optimizer_name + \
             str(with_scheduler) + '-'.join(map(str, dense))

if normalize:
    model_name += "Normalize"

##
random_seed = 5

init_epoch = 0

best_loss = sys.float_info.max

epochs = 10

batch_sizes = {
    "resnet34": 37,
    "resnet50": 95,
    "resnet101": 10
}

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
    if normalize:
        array_train.append(Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        array.append(Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    array_train.append(RandomVerticalFlip(0.5))
    array_train.append(RandomHorizontalFlip(0.5))
    array_train.append(RandomRotation((0, 360)))

    preprocess_train = Compose(array_train)

    preprocess = Compose(array)

    train_dataset = CustomDataset(labels_df, images_folder, transform=preprocess_train)
    validation_dataset = CustomDataset(labels_df, images_folder, folder="validation", transform=preprocess)
    test_dataset = CustomDataset(labels_df, images_folder, folder="test", transform=preprocess)

    return DataLoader(train_dataset, batch_size=batch_sizes.get(backbone_name, 10), shuffle=True), \
           DataLoader(validation_dataset, batch_size=batch_sizes.get(backbone_name, 10), shuffle=True), \
           DataLoader(test_dataset, batch_size=batch_sizes.get(backbone_name, 10), shuffle=True)


##

def construct_model(pretrained=True):
    backbone = getattr(models, backbone_name)(pretrained=pretrained)
    if freeze:
        for name, param in backbone.named_parameters():
            param.requires_grad = False
    if backbone_name.startswith("resnet"):
        fc = []
        for i, layer in enumerate(dense):
            fc.append((f"linear{i}", nn.Linear(backbone.fc.in_features if i == 0 else dense[i-1], layer)))
            fc.append((f"relu{i}", nn.ReLU()))

        fc.append((f'linear{len(dense)}', nn.Linear(dense[-1], num_classes)))
        fc.append((f"logsoftmax", nn.LogSoftmax(dim=1)))
        backbone.fc = nn.Sequential(OrderedDict(fc))

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
if with_scheduler:
    scheduler = optim.lr_scheduler.StepLR(optimizer, 3, gamma=0.1, verbose=True)
criterion = nn.NLLLoss()

##

if os.path.exists(model_path):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if with_scheduler:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    init_epoch = checkpoint['epoch'] + 1
    best_loss = checkpoint['best_loss']


##

images, _ = iter(train_dataloader).next()
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
                      epoch)
    writer.add_scalar('Accuracy/Validation',
                      correct * 100,
                      epoch)
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
        dict_save = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss,
        }
        if with_scheduler:
            dict_save['scheduler_state_dict'] = scheduler.state_dict()
        torch.save(dict_save, model_path)




for t in range(init_epoch, epochs):
    print(f"Epoch {t}\n-------------------------------")
    train(t)
    validation(t)
    if with_scheduler:
        scheduler.step()
print("Done!")

writer.add_hparams({
    "Preprocessing": preprocessing,
    "Data augmentation strategy": strategy,
    "Backbone": backbone_name,
    "Weights Frozen": freeze,
    "Learning rate": lr,
    "Optimizer": optimizer_name,
    "Using Scheduler": with_scheduler,
    "Input Size": input_size,
    "Fully Connected ": '-'.join(map(str, dense)),
    "Normalize": normalize
}, {
    "Best Loss": best_loss
})


writer.flush()
writer.close()
