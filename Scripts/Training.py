##
import sys
import torch
import os
import numpy as np
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from Models import ModelGenerator
from BalancedStrategies import BalancedStrategiesGenerator
from Utils import build_data_loaders, construct_optimizer

##


normalize = False
preprocessing = "denoising"
input_size = 540
strategy = "strategy4"
lr = 1e-3
optimizer_name = "Adam"
with_scheduler = False
weights = False

##
random_seed = 5
init_epoch = 0
best_loss = sys.float_info.max
epochs = 15
batch_size = 20
num_classes = 5
device = "cuda" if torch.cuda.is_available() else "cpu"
history = False
useSaved = False

##

database_folder = os.path.join("..", "Database")
images_folder = os.path.join(database_folder, "images")
annotations_file = os.path.join(database_folder, "labels",
                                f"labelsPreprocessing{preprocessing.capitalize()}.csv")
run = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

##

balancedStrategiesGenerator = BalancedStrategiesGenerator(annotations_file, random_seed)
labels_df = balancedStrategiesGenerator.apply_strategy(strategy)

##

train_dataloader, validation_dataloader, _ = build_data_loaders(preprocessing, input_size, normalize, batch_size,
                                                                labels_df, images_folder)
modelGenerator = ModelGenerator(device, num_classes)
# model, model_name = modelGenerator.resnet("resnet50", True, False, [1024, 512, 256])  # modelGenerator.li2019(1)
# model, model_name = modelGenerator.li2019(2)
model, model_name = modelGenerator.ghosh2017()

optimizer = construct_optimizer(model, optimizer_name, lr)

if with_scheduler:
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.1, verbose=True)

weights_array = torch.FloatTensor(compute_class_weight(class_weight="balanced", classes=[0, 1, 2, 3, 4],
                                                       y=labels_df["level"])).to(device)
criterion = nn.NLLLoss(weight=None if not weights
else weights_array)

##

model_whole_name = preprocessing + str(input_size) + strategy + model_name + str(lr) + optimizer_name + \
                   str(with_scheduler)

if normalize:
    model_whole_name += "Normalize"

if weights:
    model_whole_name += "Weights"

model_path = os.path.join(database_folder, "models", model_whole_name + ".pt")
##

if os.path.exists(model_path) and useSaved:
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if with_scheduler:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    init_epoch = checkpoint['epoch'] + 1
    best_loss = checkpoint['best_loss']
    run = checkpoint['run']

writer = SummaryWriter(os.path.join(database_folder, "runs", run)) if history else None

if writer and (not useSaved or not os.path.exists(model_path)):
    writer.add_hparams({
        "Preprocessing": preprocessing,
        "Data augmentation strategy": strategy,
        "Model Name": model_name,
        "Learning rate": lr,
        "Optimizer": optimizer_name,
        "Using Scheduler": with_scheduler,
        "Input Size": input_size,
        "Normalize": normalize,
        "Weights": weights,

    }, {
        "Best Loss": best_loss
    })


##


def write_scalars(dataset, metrics, x):
    writer.add_scalar(f'Loss/{dataset}', metrics["loss"], x)
    writer.add_scalar(f'Accuracy/{dataset}', metrics["accuracy"], x)
    writer.add_scalar(f"Precision/{dataset}/Average", metrics["macro avg"]["precision"], x)
    writer.add_scalar(f"Recall/{dataset}/Average", metrics["macro avg"]["recall"], x)
    writer.add_scalar(f"F1_Score/{dataset}/Average", metrics["macro avg"]["f1-score"], x)
    for j in range(5):
        writer.add_scalar(f'Precision/{dataset}/Class_{str(j)}', metrics[str(j)]["precision"], x)
        writer.add_scalar(f'Recall/{dataset}/Class_{str(j)}', metrics[str(j)]["recall"], x)
        writer.add_scalar(f'F1_Score/{dataset}/Class_{str(j)}', metrics[str(j)]["f1-score"], x)


##

def train(epoch):
    model.train()
    size = len(train_dataloader.dataset)
    number_batches = len(train_dataloader)
    running_loss = 0.0
    running_predictions, running_labels = np.array([]), np.array([])
    for i, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_labels = np.concatenate([running_labels, y.to("cpu").numpy()])
        running_predictions = np.concatenate([running_predictions, pred.argmax(1).to("cpu").numpy()])
        if i % (int(number_batches / 10)) == 0 and i != 0:
            print(f"loss: {running_loss / (int(number_batches / 10)):>7f}  [{i * len(X):>5d}/{size:>5d}]")
            metrics = classification_report(running_labels, running_predictions, output_dict=True,
                                            labels=[0, 1, 2, 3, 4])
            metrics["loss"] = running_loss / (int(number_batches / 10))
            if history:
                write_scalars("Training", metrics, epoch * len(train_dataloader) + i)
            running_loss, correct = 0.0, 0
            running_predictions, running_labels = np.array([]), np.array([])


def validation():
    model.eval()
    num_batches = len(validation_dataloader)
    test_loss = 0
    all_predictions = np.array([])
    all_labels = np.array([])
    with torch.no_grad():
        for X, y in validation_dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += criterion(pred, y).item()
            all_labels = np.concatenate([all_labels, y.to("cpu").numpy()])
            all_predictions = np.concatenate([all_predictions, pred.argmax(1).to("cpu").numpy()])
    print(f"Validation Error: \n Avg loss: {test_loss:>8f} \n")
    metrics = classification_report(all_labels, all_predictions, output_dict=True, labels=[0, 1, 2, 3, 4])
    metrics["loss"] = test_loss / num_batches
    return metrics


for t in range(init_epoch, epochs):
    print(f"Epoch {t}\n-------------------------------")
    train(t)
    validation_metrics = validation()
    if history:
        write_scalars("Validation", validation_metrics, t)
    if validation_metrics["loss"] < best_loss:
        best_loss = validation_metrics["loss"]
        if history:
            dict_save = {
                'epoch': t,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
                'run': run
            }
            if with_scheduler:
                dict_save['scheduler_state_dict'] = scheduler.state_dict()
            torch.save(dict_save, model_path)
    elif with_scheduler:
        scheduler.step()

print("Done!")

writer.flush()
writer.close()
