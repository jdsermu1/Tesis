##

import os
import numpy as np
import pandas as pd
import glob

##
# Join trainLabels.csv and testLabels.csv and create labels.csv. All annotation should appear in this file even if some
# images are damaged and preprocessing operations do not apply to them, in case this happens this should be noted in the
# file corresponding the preprocessing made.

# labels_folder = os.path.join("Database", "labels")
#
# df_train_labels = pd.read_csv(os.path.join(labels_folder, "trainLabels.csv"))
# df_test_labels = pd.read_csv(os.path.join(labels_folder, "testLabels.csv"))
#
# df_labels = pd.concat([df_test_labels.drop(columns=["Usage"]), df_train_labels])
#
# print(f'Todas los nombres de imagenes son unicos: {len(pd.unique(df_labels["image"])) == len(df_labels)}')
#
# df_labels.to_csv(os.path.join(labels_folder, "labels.csv"), index=False)

##
# Find out distribution of labels in classes

# labels = pd.read_csv(os.path.join("Database", "labels", "labels.csv"))
#
# stats = pd.DataFrame({"percentage": labels["level"].value_counts(normalize=True)*100,
#                       "count": labels["level"].value_counts()})
#
# stats


##
# Compare results Li 2019 get recall and precision

matrix_nn1 = np.array([[39031, 2693, 2361, 75, 160],
                       [118, 438, 185, 0, 1],
                       [339, 626, 5058, 738, 392],
                       [0, 0, 170, 332, 102],
                       [43, 5, 85, 69, 551]])

recall_nn1 = np.diagonal(np.divide(matrix_nn1, matrix_nn1.sum(0, keepdims=True)))

precision_nn1 = np.diagonal(np.divide(matrix_nn1, matrix_nn1.sum(1, keepdims=True)))

print("Precision", precision_nn1, "Recall", recall_nn1)