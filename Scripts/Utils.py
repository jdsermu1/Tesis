import os
import pandas as pd
from skimage import io
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Compose, Resize, Normalize, RandomHorizontalFlip, \
    RandomVerticalFlip, RandomRotation, CenterCrop
from torch import optim
import glob


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


def build_data_loaders(preprocessing, input_size, normalize, batch_size, labels_df, images_folder):
    array_train = [ToTensor()]
    array = [ToTensor()]
    if preprocessing in ["original"]:
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

    return DataLoader(train_dataset, batch_size=batch_size,  shuffle=True, num_workers=2, persistent_workers=True), \
           DataLoader(validation_dataset, batch_size=batch_size, num_workers=2, persistent_workers=True),\
           DataLoader(test_dataset, batch_size=batch_size, num_workers=2, persistent_workers=True)


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
    labels = pd.read_csv(os.path.join('..', "Database", "labels", "labels.csv"))
    if not preprocessing:
        return labels
    elif os.path.exists(os.path.join("..", "Database", "preprocessing images", preprocessing)):
        list_dir = glob.glob(os.path.join("..", "Database", "preprocessing images", preprocessing, "**.jpeg"))
        list_dir = [i[i.rfind(os.path.sep) + 1:-5] for i in list_dir]
        labels_preprocessing = labels[labels["image"].isin(list_dir)].copy()
        return labels_preprocessing
    else:
        raise Exception(f"Preprocessing '{preprocessing}' was not found")
