##

import os
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil
from progressbar import ProgressBar

##

preprocessing = "adaptation"
strategy = "strategy2"

# strategy1 : 100 validation images per category, 312 test images per category, the rest of images are taken for
# training without any augmentation strategy.
# strategy2 : 100 validation images per category, 312 test images per category,1502 train images pero category
##

labels_folder = os.path.join("..", "Database", "labels")
labels = pd.read_csv(os.path.join(labels_folder, f"labelsPreprocessing{preprocessing.capitalize()}.csv"))


##


def strategy1():
    size_validation = 100
    size_test = 312
    size_partition = size_test + size_validation
    labels_strategy = pd.DataFrame({})
    for c in range(5):
        labels_c = labels[labels['level'] == c]
        labels_train, labels_test_val = train_test_split(labels_c, test_size=size_partition)
        labels_train["set"] = "train"
        labels_test, labels_validation = train_test_split(labels_test_val, test_size=size_validation)
        labels_test["set"] = "test"
        labels_validation["set"] = "validation"
        labels_strategy = pd.concat([labels_strategy, labels_validation, labels_test, labels_train])
    return labels_strategy
##


def strategy2():
    size_validation = 100
    size_test = 312
    size_train = 1502
    size_partition = size_test + size_validation
    labels_strategy = pd.DataFrame({})
    for c in range(5):
        labels_c = labels[labels['level'] == c]
        labels_train, labels_test_val = train_test_split(labels_c, test_size=size_partition)
        if len(labels_train) > size_train:
            labels_train, _ = train_test_split(labels_train, train_size=size_train)
        labels_train["set"] = "train"
        labels_test, labels_validation = train_test_split(labels_test_val, test_size=size_validation)
        labels_test["set"] = "test"
        labels_validation["set"] = "validation"
        labels_strategy = pd.concat([labels_strategy, labels_validation, labels_test, labels_train])
    return labels_strategy


##


if strategy == "strategy1":
    func = strategy1
else:
    func = strategy1
final_labels = func()
final_labels.to_csv(os.path.join(labels_folder,
                                 f"labelsPreprocessing{preprocessing.capitalize()}{strategy.capitalize()}.csv"),
                    index=False)
