##
import sys
import pandas as pd
import torch
import os
import numpy as np
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from Scripts.BalancedStrategies import BalancedStrategiesGenerator
from Scripts.Models import ModelGenerator
from Scripts.UtilsMetacost import metacost_validation, CostMatrixGenerator
from Scripts.Utils import build_data_loaders, construct_optimizer, write_scalars, train, validation
import gc
import dask.dataframe as dd
from dask.multiprocessing import get

##
# Recording parameters

execution_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
base_run = None
run = base_run if base_run else execution_time
recordProgress = False

##
# Metacost parameters
random_seed = 5
frac = 1
m = 10
q = True

##
# Training parameters
epochs = 10
train_batch_size = 15
eval_batch_size = 90
device = "cuda" if torch.cuda.is_available() else "cpu"
preprocessing = "adaptation"
lr = 1e-3
optimizer_name = "Adam"
classification_type = "ordinal"   # categorical, ordinal, special_ordinal
assert(classification_type in ["categorical", "ordinal", "special_ordinal"])

##
# Folders
database_folder = os.path.join("..", "Database")
images_folder = os.path.join(database_folder, "preprocessing images", preprocessing)
metacost_folder = os.path.join(database_folder, "metacost")
run_folder = os.path.join(metacost_folder, run)

##
# Labels preparation
balancedStrategiesGenerator = BalancedStrategiesGenerator(preprocessing, random_seed)
labels = balancedStrategiesGenerator.apply_strategy("strategy0")


##
# Assistance functions


def create_samples_models_train():
    if os.path.exists(os.path.join(run_folder, "metacostLabels.csv")):
        labels_metacost = pd.read_csv(os.path.join(run_folder, "metacostLabels.csv"))
        labels_metacost = labels_metacost[labels_metacost["model"].isin([f"model{str(i)}" for i in range(0, m)])]
        print(f"Metacost labels found and loaded from directory")
    else:
        labels_metacost = pd.DataFrame([])
    for i in range(0, m):
        if "model" not in labels_metacost or f"model{str(i)}" not in pd.unique(labels_metacost["model"]):

            train_labels = labels[labels["set"] == "train"].sample(frac=frac, replace=True,
                                                                   random_state=random_seed + i)
            iteration_labels = pd.concat([train_labels, labels[labels["set"] == "validation"]]).copy()
            train_dataloader = build_data_loaders(preprocessing, 540, False, train_batch_size, iteration_labels,
                                                  images_folder, folder="train", num_workers=2,
                                                  classification_type=classification_type)
            validation_dataloader = build_data_loaders(preprocessing, 540, False, eval_batch_size, iteration_labels,
                                                       images_folder, folder="validation", num_workers=4,
                                                       classification_type=classification_type)
            model, _ = modelGenerator.resnet("resnet50", True, False, [1024, 512, 256],
                                             classification_type=classification_type)
            optimizer = construct_optimizer(model, optimizer_name, lr)
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1, verbose=True)
            print(f"Training of model {i} began")
            if recordProgress and not os.path.exists(os.path.join(run_folder, "models", f"model{str(i)}")):
                os.mkdir(os.path.join(run_folder, "models", f"model{str(i)}"))
                print(f"Folder for model {i} was created")
            writer = SummaryWriter(os.path.join(run_folder, "models", f"model{str(i)}", "log")) if \
                recordProgress else None
            model_training(model, optimizer, scheduler, train_dataloader, validation_dataloader, index=i, writer=writer)
            print(f"Model was trained")
            aux_df = labels[labels["set"] == "train"].copy()
            aux_df["model"] = f"model{str(i)}"
            aux_df["training"] = np.where(aux_df["image"].isin(pd.unique(train_labels["image"])), True, False)
            probabilities_dataloader = build_data_loaders(preprocessing, 540, False, eval_batch_size,
                                                          aux_df, images_folder, folder="train",
                                                          num_workers=4)
            aux_df.drop(columns="set", inplace=True)
            aux_df = aux_df.merge(metacost_validation(model, probabilities_dataloader, device), on="image", how="inner")
            gc.collect()
            torch.cuda.empty_cache()
            labels_metacost = pd.concat([labels_metacost, aux_df])
            if recordProgress:
                print(f"Copy of metacost labels was saved")
                labels_metacost.to_csv(os.path.join(run_folder, "metacostLabels.csv"), index=False)
        else:
            print(f"Training of model {i} was skipped, labels for model were found")
    return labels_metacost


def model_training(model, optimizer, scheduler, train_dataloader, validation_dataloader, index, writer=None):
    best_loss = sys.float_info.max
    init_epoch = 0
    checkpoint_path = os.path.join(run_folder, "models", f"model{str(index)}", "checkpoint.pt")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        best_loss_epoch = checkpoint['best_loss_epoch']
        init_epoch = best_loss_epoch + 1
        best_loss = checkpoint['best_loss']
        print(f"Model {index} loaded from checkpoint, starting epoch: {init_epoch}")
    for epoch in range(init_epoch, epochs):
        train(model, optimizer, criterion, train_dataloader, epoch, recordProgress, writer, device, classification_type)
        gc.collect()
        torch.cuda.empty_cache()
        validation_metrics = validation(model, criterion, validation_dataloader, device, classification_type)
        gc.collect()
        torch.cuda.empty_cache()
        if recordProgress:
            write_scalars(writer, "Validation", validation_metrics, epoch)
            print(f"Validation step recorded")
        if validation_metrics["loss"] < best_loss:
            best_loss = validation_metrics["loss"]
            best_loss_epoch = epoch
            print(f"New best loss {best_loss} in epoch {best_loss_epoch}")
            if recordProgress:
                dict_save = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_loss': best_loss,
                    "best_loss_epoch": best_loss_epoch,
                    'scheduler_state_dict': scheduler.state_dict()
                }
                torch.save(dict_save, checkpoint_path)
                print(f"Checkpoint recorded")
        else:
            scheduler.step()
    if recordProgress:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("Best loss model version recovered")
    else:
        print("WARNING: Process is not being recorded so no checkpoint from best loss model version is being recovered")


##
# Algorithm
modelGenerator = ModelGenerator(device, 5)
if classification_type == "categorical":
    criterion = nn.NLLLoss()
else:
    criterion = nn.MSELoss(reduction='sum')

if recordProgress and not os.path.exists(run_folder):
    os.makedirs(os.path.join(run_folder, "models"))
    print(f"Folder for run: {run} created as subdirectory to save models")

df_labels_metacost = create_samples_models_train()


##

df_adjusted_labels = labels[labels["set"] == "train"]

costMatrixGenerator = CostMatrixGenerator(df_adjusted_labels, seed=random_seed)
cost_matrix = costMatrixGenerator.distance_value()
print(cost_matrix)


def find_new_level(row: pd.Series):
    df_votes = df_labels_metacost[df_labels_metacost["image"] == row["image"]].copy()
    if not q:
        df_votes = df_votes[df_votes["training"] is False]
    estimated_p = df_votes[["P" + str(j) for j in range(5)]].mean().to_numpy()
    estimated_cost = np.matmul(cost_matrix, estimated_p.reshape(estimated_p.shape[0], 1))
    estimated_category = np.argmin(estimated_cost)
    return estimated_category


df_adjusted_labels["new_level"] = dd.from_pandas(df_adjusted_labels,
                                                 npartitions=22).map_partitions(
    lambda df: df.apply(find_new_level, axis=1)).compute(scheduler=get)
print(pd.DataFrame({"percentage": df_adjusted_labels["new_level"].value_counts(normalize=True)*100}))
