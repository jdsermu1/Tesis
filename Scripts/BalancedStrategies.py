import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from Utils import get_labels


class BalancedStrategiesGenerator:
    def __init__(self, preprocessing, random_seed):
        self.annotations = get_labels(preprocessing=preprocessing)
        self.random_seed = random_seed


    def strategy1(self, labels):
        size_validation = 100
        size_test = 312
        size_partition = size_test + size_validation
        labels_strategy = pd.DataFrame({})
        for c in range(5):
            labels_c = labels[labels['level'] == c]
            labels_train, labels_test_val = train_test_split(labels_c, test_size=size_partition,
                                                             random_state=self.random_seed)
            labels_train["set"] = "train"
            labels_test, labels_validation = train_test_split(labels_test_val, test_size=size_validation,
                                                              random_state=self.random_seed)
            labels_test["set"] = "test"
            labels_validation["set"] = "validation"
            labels_strategy = pd.concat([labels_strategy, labels_validation, labels_test, labels_train])
        return labels_strategy

    def strategy2(self, labels):
        size_validation = 100
        size_test = 312
        size_train = 1502
        size_partition = size_test + size_validation
        labels_strategy = pd.DataFrame({})
        for c in range(5):
            labels_c = labels[labels['level'] == c]
            labels_train, labels_test_val = train_test_split(labels_c, test_size=size_partition,
                                                             random_state=self.random_seed)
            if len(labels_train) > size_train:
                labels_train, _ = train_test_split(labels_train, train_size=size_train, random_state=self.random_seed)
            labels_train["set"] = "train"
            labels_test, labels_validation = train_test_split(labels_test_val, test_size=size_validation,
                                                              random_state=self.random_seed)
            labels_test["set"] = "test"
            labels_validation["set"] = "validation"
            labels_strategy = pd.concat([labels_strategy, labels_validation, labels_test, labels_train])
        return labels_strategy


    def strategy3(self, labels):
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
                                                                 random_state=self.random_seed)
            labels_validation_c, labels_test_c = train_test_split(labels_test_val_c, test_size=size_test,
                                                                  random_state=self.random_seed)
            labels_train = pd.concat([labels_train, labels_train_c])
            labels_test = pd.concat([labels_test, labels_test_c])
            labels_validation = pd.concat([labels_validation, labels_validation_c])

        rus = RandomUnderSampler(sampling_strategy={0: size_train}, random_state=self.random_seed)
        ros = RandomOverSampler(sampling_strategy="not majority", random_state=self.random_seed)
        labels_train, _ = rus.fit_sample(labels_train, labels_train["level"])
        labels_train, _ = ros.fit_sample(labels_train, labels_train["level"])
        labels_train["set"] = "train"
        labels_test["set"] = "test"
        labels_validation["set"] = "validation"
        labels_strategy = pd.concat([labels_train, labels_validation, labels_test])
        return labels_strategy

    def strategy4(self, labels):
        test_size = 15600
        validation_size = 5000
        train_size = 15000
        labels_train, labels_test_validation = train_test_split(labels, test_size=test_size + validation_size,
                                                                stratify=labels["level"], random_state=self.random_seed)
        labels_validation, labels_test = train_test_split(labels_test_validation, test_size=test_size,
                                                          stratify=labels_test_validation["level"],
                                                          random_state=self.random_seed)
        rus = RandomUnderSampler(sampling_strategy={0: train_size}, random_state=self.random_seed)
        ros = RandomOverSampler(sampling_strategy="not majority", random_state=self.random_seed)
        labels_train, _ = rus.fit_sample(labels_train, labels_train["level"])
        labels_train, _ = ros.fit_sample(labels_train, labels_train["level"])
        labels_train["set"] = "train"
        labels_validation["set"] = "validation"
        labels_test["set"] = "test"
        labels_strategy = pd.concat([labels_train, labels_validation, labels_test])
        return labels_strategy



    def apply_strategy(self, strategy):
        if strategy == "strategy1":
            func = self.strategy1
        elif strategy == "strategy2":
            func = self.strategy2
        elif strategy == "strategy3":
            func = self.strategy3
        elif strategy == "strategy4":
            func = self.strategy4
        else:
            func = self.strategy1
        labels_df = func(self.annotations)
        return labels_df
