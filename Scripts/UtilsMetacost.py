import pandas as pd
import torch


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

