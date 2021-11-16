import os
import pandas as pd
import torch
from skimage import io
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor, Compose, Resize, Normalize, RandomHorizontalFlip, \
    RandomVerticalFlip, RandomRotation, CenterCrop
from torch import optim
import glob
import numpy as np


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
        return image, label, self.img_labels.iloc[idx, 0]


def build_data_loaders(preprocessing, input_size, normalize, batch_size, labels_df, images_folder,
                       classification_type: str = "categorical", folder="train", num_workers=2):
    array = [ToTensor()]
    if preprocessing in ["original"]:
        array.append((CenterCrop((540, 540))))
    if input_size != 540:
        array.append(Resize(input_size))
    if normalize:
        array.append(Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    if folder == "train":
        array.append(RandomVerticalFlip(0.5))
        array.append(RandomHorizontalFlip(0.5))
        array.append(RandomRotation((0, 360)))

    preprocess = Compose(array)

    target_transform = None
    if classification_type == "ordinal":
        # TODO: Turn target into double for classification
        target_transform = target_to_ordinal
    elif classification_type == "ordinal_special":
        target_transform = target_to_special_ordinal

    dataset = CustomDataset(labels_df, images_folder, folder=folder, transform=preprocess,
                            target_transform=target_transform)

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=True)


def construct_optimizer(m, optimizer_name, lr):
    if optimizer_name == "Adam":
        opt = optim.Adam(m.parameters(), lr=lr)
    elif optimizer_name == "SGD":
        opt = optim.SGD(m.parameters(), lr=lr)
    elif optimizer_name == "RMSprop":
        opt = optim.RMSprop(m.parameters(), lr=lr)
    else:
        opt = optim.SGD(m.parameters(), lr=lr)
    return opt


def get_labels(preprocessing=None):
    labels = pd.read_csv(os.path.join("..", "Database", "labels", "labels.csv"))
    if not preprocessing:
        return labels
    elif os.path.exists(os.path.join("..", "Database", "preprocessing images", preprocessing)):
        list_dir = glob.glob(os.path.join("..", "Database", "preprocessing images", preprocessing, "**.jpeg"))
        list_dir = [i[i.rfind(os.path.sep) + 1:-5] for i in list_dir]
        labels_preprocessing = labels[labels["image"].isin(list_dir)].copy()
        return labels_preprocessing
    else:
        raise Exception(f"Preprocessing '{preprocessing}' was not found")


def target_to_special_ordinal(target):
    aux_tensor = torch.tensor([0, 1, 2, 3, 4])
    return torch.where(aux_tensor <= target, 1.0, 0.0)


def target_to_ordinal(target):
    return torch.tensor(target, requires_grad=False).type(torch.FloatTensor)


def write_scalars(summary_writer: SummaryWriter, dataset: str, metrics: dict, x):
    summary_writer.add_scalar(f'Loss/{dataset}', metrics["loss"], x)
    summary_writer.add_scalar(f'Accuracy/{dataset}', metrics["accuracy"], x)
    summary_writer.add_scalar(f"Precision/{dataset}/Average", metrics["macro avg"]["precision"], x)
    summary_writer.add_scalar(f"Recall/{dataset}/Average", metrics["macro avg"]["recall"], x)
    summary_writer.add_scalar(f"F1_Score/{dataset}/Average", metrics["macro avg"]["f1-score"], x)
    for j in range(5):
        summary_writer.add_scalar(f'Precision/{dataset}/Class_{str(j)}', metrics[str(j)]["precision"], x)
        summary_writer.add_scalar(f'Recall/{dataset}/Class_{str(j)}', metrics[str(j)]["recall"], x)
        summary_writer.add_scalar(f'F1_Score/{dataset}/Class_{str(j)}', metrics[str(j)]["f1-score"], x)


def train(model, optimizer, criterion, dataloader, epoch, record_progress, writer, device, classification_type):
    model.train()
    size = len(dataloader.dataset)
    number_batches = len(dataloader)
    running_loss = 0.0
    running_predictions, running_labels = np.array([]), np.array([])
    for i, (X, y, _) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if classification_type == "categorical":
            running_labels = np.concatenate([running_labels, y.to("cpu").numpy()])
            running_predictions = np.concatenate([running_predictions, pred.argmax(1).to("cpu").numpy()])
        elif classification_type == "special_ordinal":
            running_labels = np.concatenate([running_labels, ((y.to("cpu") > 0.5).cumprod(axis=1)
                                                              .sum(axis=1)-1).numpy()])
            running_predictions = np.concatenate([running_predictions, ((pred.to("cpu") > 0.5).cumprod(axis=1)
                                                                        .sum(axis=1)-1).numpy()])
        elif classification_type == "ordinal":
            running_labels = np.concatenate([running_labels, y.to("cpu").numpy()])
            running_predictions = np.concatenate([running_predictions, torch.round(pred.to("cpu")).detach().numpy()[:,0]])
        if i % (int(number_batches / 3)) == 0 and i != 0:
            print(f"loss: {running_loss / (int(number_batches / 5)):>7f}  [{i * len(X):>5d}/{size:>5d}]")
            metrics = classification_report(running_labels, running_predictions, output_dict=True,
                                            labels=[0, 1, 2, 3, 4])
            metrics["loss"] = running_loss / (int(number_batches / 10))
            if record_progress:
                if not metrics.get("accuracy"):
                    metrics["accuracy"] = accuracy_score(running_labels, running_predictions, normalize=True)
                write_scalars(writer, "Training", metrics, epoch * len(dataloader) + i)
                print(f"Training step recorded")
            running_loss, correct = 0.0, 0
            running_predictions, running_labels = np.array([]), np.array([])


def validation(model, criterion, dataloader, device, classification_type):
    model.eval()
    num_batches = len(dataloader)
    test_loss = 0
    all_predictions = np.array([])
    all_labels = np.array([])
    with torch.no_grad():
        for i, (X, y, _) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss = criterion(pred, y).item()
            if classification_type == "categorical":
                all_labels = np.concatenate([all_labels, y.to("cpu").numpy()])
                all_predictions = np.concatenate([all_predictions, pred.argmax(1).to("cpu").numpy()])
            elif classification_type == "special_ordinal":
                all_labels = np.concatenate(
                    [all_labels, ((y.to("cpu") > 0.5).cumprod(axis=1).sum(axis=1) - 1).numpy()])
                all_predictions = np.concatenate(
                    [all_predictions, ((pred.to("cpu") > 0.5).cumprod(axis=1).sum(axis=1) - 1).numpy()])
            elif classification_type == "ordinal":
                all_labels = np.concatenate([all_labels, y.to("cpu").numpy()])
                all_predictions = np.concatenate([all_predictions, torch.round(pred).to("cpu").numpy()[:,0]])
    print(f"Validation Error Avg loss: {test_loss:>8f}")
    metrics = classification_report(all_labels, all_predictions, output_dict=True, labels=[0, 1, 2, 3, 4])
    if not metrics.get("accuracy"):
        metrics["accuracy"] = accuracy_score(all_labels, all_predictions, normalize=True)
    metrics["loss"] = test_loss / num_batches
    return metrics
