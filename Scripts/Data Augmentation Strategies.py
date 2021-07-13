##

import os
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil
from progressbar import ProgressBar
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

##

preprocessing = "adaptation"
strategy = "strategy4"

# strategy1 : 100 validation images per category, 312 test images per category, the rest of images are taken for
# training without any augmentation strategy.
# strategy2 : 100 validation images per category, 312 test images per category,1502 train images pero category
# strategy3 : 100 validation images per category, 312 test images per category, the remaining images got to training
# but the majority class is undersampled to 15000 images and the minority classes are oversampled to 15000 images
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


def strategy3():
    size_validation = 100
    size_test = 312
    size_train = 15000
    size_partition = size_validation + size_test
    labels_train = pd.DataFrame({})
    labels_test = pd.DataFrame({})
    labels_validation = pd.DataFrame({})
    for c in range(5):
        labels_c = labels[labels['level'] == c]
        labels_train_c, labels_test_val_c = train_test_split(labels_c, test_size=size_partition)
        labels_validation_c, labels_test_c = train_test_split(labels_test_val_c, test_size=size_test)
        labels_train = pd.concat([labels_train, labels_train_c])
        labels_test = pd.concat([labels_test, labels_test_c])
        labels_validation = pd.concat([labels_validation, labels_validation_c])


    rus = RandomUnderSampler(sampling_strategy={0: size_train})
    ros = RandomOverSampler(sampling_strategy="not majority")
    labels_train, _ = rus.fit_sample(labels_train, labels_train["level"])
    labels_train, _ = ros.fit_sample(labels_train, labels_train["level"])
    labels_train["set"] = "train"
    labels_test["set"] = "test"
    labels_validation["set"] = "validation"
    labels_strategy = pd.concat([labels_train, labels_validation, labels_test])
    return labels_strategy

##


def strategy4():
    test_size = 15600
    validation_size = 5000
    train_size = 15000
    labels_train, labels_test_validation = train_test_split(labels, test_size=test_size+validation_size,
                                                            stratify=labels["level"])
    labels_validation, labels_test = train_test_split(labels_test_validation, test_size=test_size,
                                                      stratify=labels_test_validation["level"])
    rus = RandomUnderSampler(sampling_strategy={0: train_size})
    ros = RandomOverSampler(sampling_strategy="not majority")
    labels_train, _ = rus.fit_sample(labels_train, labels_train["level"])
    labels_train, _ = ros.fit_sample(labels_train, labels_train["level"])
    labels_train["set"] = "train"
    labels_validation["set"] = "validation"
    labels_test["set"] = "test"
    labels_strategy = pd.concat([labels_train, labels_validation, labels_test])
    return labels_strategy


##


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
final_labels = func()
final_labels.to_csv(os.path.join(labels_folder,
                                 f"labelsPreprocessing{preprocessing.capitalize()}{strategy.capitalize()}.csv"),
                    index=False)
