##
import sys
import torch
import os
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight
from Models import ModelGenerator
from BalancedStrategies import BalancedStrategiesGenerator
from Utils import build_data_loaders, construct_optimizer, write_scalars, train, validation

##


normalize = False
preprocessing = "adaptationDenoising2"
input_size = 540
strategy = "strategy4"
lr = 1e-3
optimizer_name = "Adam"
with_scheduler = True
weights = False


##
random_seed = 5
init_epoch = 0
best_loss = sys.float_info.max
epochs = 18
train_batch_size = 10
eval_batch_size = 90
num_classes = 5
device = "cuda" if torch.cuda.is_available() else "cpu"
history = False
useSaved = False
classification_type = "ordinal"   # categorical, ordinal, special_ordinal

assert(classification_type in ["categorical", "ordinal", "special_ordinal"])

##

database_folder = os.path.join("..", "Database")
images_folder = os.path.join(database_folder, "preprocessing images", preprocessing)
run = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

##

balancedStrategiesGenerator = BalancedStrategiesGenerator(preprocessing, random_seed)
labels_df = balancedStrategiesGenerator.apply_strategy(strategy)

##

train_dataloader = build_data_loaders(preprocessing, input_size, normalize, train_batch_size, labels_df, images_folder,
                                      folder="train", num_workers=2, classification_type=classification_type)
validation_dataloader = build_data_loaders(preprocessing, input_size, normalize, train_batch_size, labels_df,
                                           images_folder, folder="train", num_workers=4,
                                           classification_type=classification_type)
modelGenerator = ModelGenerator(device, num_classes)
model, model_name = modelGenerator.resnet("resnet50", True, False, [1024, 512, 256],
                                          classification_type=classification_type)  # modelGenerator.li2019
# model, model_name = modelGenerator.li2019(2)
# model, model_name = modelGenerator.ghosh2017()

optimizer = construct_optimizer(model, optimizer_name, lr)

scheduler = None
if with_scheduler:
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1, verbose=True)

weights_array = torch.FloatTensor(compute_class_weight(class_weight="balanced", classes=[0, 1, 2, 3, 4],
                                                       y=labels_df["level"])).to(device)

if classification_type == "categorical":
    criterion = nn.NLLLoss(weight=None if not weights else weights_array)
else:
    criterion = nn.MSELoss(reduction='sum')


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

##


for t in range(init_epoch, epochs):
    print(f"Epoch {t}\n-------------------------------")
    train(model, optimizer, criterion, train_dataloader, t, history, writer, device, classification_type)
    validation_metrics = validation(model, criterion, validation_dataloader, device, classification_type)
    if history:
        write_scalars(writer, "Validation", validation_metrics, t)
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

if history:
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

writer.flush()
writer.close()
