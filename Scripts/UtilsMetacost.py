import pandas as pd
import torch
import numpy as np


class CostMatrixGenerator:

    def __init__(self, df_labels, seed):
        self.df_probabilities = pd.DataFrame({"percentage": df_labels["level"].value_counts(normalize=True)*100})
        self.rng = np.random.default_rng(seed)


    def random_value(self, categories=5, max_value_i_i=1000, max_value_i_j=10000):
        np_cost_matrix = self.rng.random([categories, categories])
        for i in range(np_cost_matrix.shape[0]):
            for j in range(np_cost_matrix.shape[1]):
                max_value = max_value_i_i if i == j else max_value_i_j
                np_cost_matrix[i, j] *= max_value
        return np_cost_matrix

    def random_frequency_value(self, categories=5, max_value_i_i=1000, max_value_i_j=2000):
        np_cost_matrix = self.rng.random([categories, categories])
        for i in range(np_cost_matrix.shape[0]):
            for j in range(np_cost_matrix.shape[1]):
                if i == j:
                    np_cost_matrix[i, j] *= max_value_i_i
                else:
                    np_cost_matrix[i, j] *= max_value_i_j * self.df_probabilities.loc[i] / self.df_probabilities.loc[j]
        return np_cost_matrix


    def frequency_value(self, categories=5, max_value=1000):
        np_cost_matrix = np.ones([categories, categories])
        for i in range(np_cost_matrix.shape[0]):
            for j in range(np_cost_matrix.shape[1]):
                if i == j:
                    np_cost_matrix[i, j] = 0
                else:
                    np_cost_matrix[i, j] *= max_value * self.df_probabilities.loc[i] / self.df_probabilities.loc[j]
        return np_cost_matrix


def metacost_validation(model, dataloader, device):
    model.eval()
    all_probabilities = pd.DataFrame([])
    with torch.no_grad():
        for i, (X, y, index) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            probabilities = pd.DataFrame(data=torch.exp(pred.to("cpu")).numpy(),
                                         columns=["P" + str(j) for j in range(5)])
            probabilities["image"] = index
            all_probabilities = pd.concat([all_probabilities, probabilities], ignore_index=True)
    return all_probabilities




