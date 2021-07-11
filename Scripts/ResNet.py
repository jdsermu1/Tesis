##
import random

import numpy
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

##

num_classes = 5
batch_size = 50
epochs = 12
lr = 1e-5
preprocessing = "original"
strategy = "strategy3"

##

device = "cuda" if torch.cuda.is_available() else "cpu"

database_folder = os.path.join("..", "Database")

images_folder = os.path.join(database_folder, "preprocessing images", preprocessing)

annotations_file = os.path.join(database_folder, "labels",
                                f"labelsPreprocessing{preprocessing.capitalize()}{strategy.capitalize()}.csv")

writer = SummaryWriter(os.path.join(database_folder, "runs", datetime.now().strftime("%d-%m-%Y %H:%M:%S")))

##


class CustomDataset(Dataset):
    def __init__(self, labels_file, img_dir, folder="train", transform=None, target_transform=None):
        self.img_dir = img_dir
        self.img_labels = pd.read_csv(labels_file)
        self.img_labels = self.img_labels[self.img_labels["set"] == folder]
        self.transform = transform
        self.target_transform = target_transform
        self.folder = folder

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        label = self.img_labels.iloc[idx, 1]
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0]+".jpeg")
        image = io.imread(img_path)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


preprocess_train = Compose([
    ToTensor(),
    CenterCrop((540, 540)),
    RandomVerticalFlip(0.5),
    RandomHorizontalFlip(0.5),
    RandomRotation((0, 360)),
    # Resize(224),
    # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

preprocess = Compose([
    ToTensor(),
    CenterCrop((540, 540)),
    # RandomVerticalFlip(0.5),
    # RandomHorizontalFlip(0.5),
    # RandomRotation((0, 360), interpolation=InterpolationMode.BILINEAR)
    # Resize(224),
    # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

preprocess_target = Compose([
    ToTensor()
])

# to_one_hot = Lambda(lambda y: torch.zeros(5, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))
train_dataset = CustomDataset(annotations_file, images_folder, transform=preprocess_train)
validation_dataset = CustomDataset(annotations_file, images_folder, folder="validation", transform=preprocess)
test_dataset = CustomDataset(annotations_file, images_folder, folder="test", transform=preprocess)
##

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

##

images, labels = iter(train_dataloader).next()
img_grid = torchvision.utils.make_grid(images)
writer.add_image('train_example', img_grid)

##

model = models.resnet34(pretrained=True)

# for name, param in model.named_parameters():
#     if not name.startswith("layer4"):
#         param.requires_grad = False

model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 1024),
    nn.ReLU(),
    nn.Linear(1024, 8),
    # nn.ReLU(),
    # nn.Linear(512, num_classes),
    nn.LogSoftmax(dim=1)
)

model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=lr)
# optim.SGD(model.parameters(), lr=lr)  #
# optim.RMSprop(model.parameters(), lr=lr)

scheduler = optim.lr_scheduler.StepLR(optimizer, 3, gamma=0.1, verbose=True)

criterion = nn.NLLLoss()


##


def train(epoch):
    model.train()
    size = len(train_dataloader.dataset)
    number_batches = len(train_dataloader)
    running_loss, correct = 0.0, 0
    running_predictions, running_labels = numpy.array([]), numpy.array([])
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
        if i % (int(number_batches/10)) == 0 and i != 0:
            loss, current = loss.item(), i * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            writer.add_scalar('Loss/Training',
                              running_loss / (int(number_batches/10)),
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
            running_predictions, running_labels = numpy.array([]), numpy.array([])



def validation(epoch):
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
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    writer.add_scalar('Loss/Validation',
                      test_loss,
                      epoch+1)
    writer.add_scalar('Accuracy/Validation',
                      correct*100,
                      epoch+1)
    precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels, all_predictions, labels=[0, 1, 2,
                                                                                                                3, 4])
    for i in range(5):
        writer.add_scalar(f'Precision/Validation/Class_{str(i)}',
                          precision[i],
                          epoch + 1)
        writer.add_scalar(f'Recall/Validation/Class_{str(i)}',
                          recall[i],
                          epoch + 1)
        writer.add_scalar(f'F1_Score/Validation/Class_{str(i)}',
                          f1_score[i],
                          epoch + 1)


for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(t)
    validation(t)
    scheduler.step()
print("Done!")



writer.add_hparams({})


writer.flush()
writer.close()


