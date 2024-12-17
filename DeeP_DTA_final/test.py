
'''Loads a trained model and evaluates it on a test dataset. 
It calculates performance metrics such as Loss, CI, R², 
and RMSE and prints the results.'''

import torch
import numpy as np
from cindex_score import cindex_score
from sklearn.metrics import r2_score, mean_squared_error
import math
from config import *
from dataset import *
from model import *


def evaluate_model(model, dataloader, criterion, device):
    """Evaluates the model on a given dataset and returns metrics."""
    model.eval()
    metrics = {"loss": [], "CI": [], "R2": [], "RMSE": []}
    all_affinity = np.array([])
    y_pred = np.array([])

    for batch_i, (XD, XP, y) in enumerate(dataloader):
        drug_encode = XD.to(device)
        protein_encode = XP.to(device)
        affinity = y.to(device)

        with torch.no_grad():
            predictions = model(drug_encode, protein_encode)
            y_pred = np.append(
                y_pred, predictions.squeeze().detach().cpu().numpy(), axis=0)
            all_affinity = np.append(
                all_affinity, affinity.detach().cpu().numpy())

        loss = criterion(predictions.squeeze(), affinity).item()
        ci = cindex_score(affinity.detach().cpu().numpy(),
                          predictions.squeeze().detach().cpu().numpy())
        r2 = r2_score(all_affinity, y_pred)
        rmse = math.sqrt(mean_squared_error(all_affinity, y_pred))

        metrics["loss"].append(loss)
        metrics["CI"].append(ci)
        metrics["R2"].append(r2)
        metrics["RMSE"].append(rmse)

    # Compute mean metrics
    return {k: round(np.mean(v), 4) for k, v in metrics.items()}


def test_deep_dta(model, test_dataloader, checkpoint_model, device=DEVICE):
    model.load_state_dict(torch.load(checkpoint_model, map_location=device))
    print(f"Model loaded from {checkpoint_model}")
    model.to(device)

    criterion = torch.nn.MSELoss().to(device)

    print("Starting evaluation on the test set...\n")

    test_metrics = evaluate_model(model, test_dataloader, criterion, device)

    print(
        f"Test Metrics: Loss: {test_metrics['loss']}, CI: {test_metrics['CI']}, R2: {test_metrics['R2']}, RMSE: {test_metrics['RMSE']}")
    return test_metrics


if __name__ == "__main__":
    from config import *
    from dataset import DatasetFromCSV
    from torch.utils.data import DataLoader

    # Load data and create DataLoader
    # you should either define your test_data.csv in the config or give it's path here
    # and if you want only tes it run this file separately

    _, _, test_data = load_data(CSV_PATH)
    test_dataset = DatasetFromCSV(
        test_data, seqlen=SEQUENCE_LENGTH, smilen=SMILES_LENGTH)
    test_dataloader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Initialize model
    model = DeepDTA().to(DEVICE)

    # Test the model
    test_metrics = test_deep_dta(model, test_dataloader, checkpoint_model)
