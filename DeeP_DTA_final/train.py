import torch
import torch.optim as optim
import torch.nn as nn
import time
import sys
import numpy as np
import os
import matplotlib.pyplot as plt
from model import DeepDTA
from cindex_score import cindex_score
from config import DEVICE
from sklearn.metrics import r2_score, mean_squared_error
import math


def plot_fig(loss_values, ci_values, r2_values, rmse_values, num_epochs):
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 2, 1)
    plt.plot(range(1, num_epochs + 1), loss_values, label='Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(2, 2, 2)
    plt.plot(range(1, num_epochs + 1), ci_values, label='CI', color='orange')
    plt.title('CI over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('CI')

    plt.subplot(2, 2, 3)
    plt.plot(range(1, num_epochs + 1), r2_values, label='R2', color='green')
    plt.title('R2 over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('R2')

    plt.subplot(2, 2, 4)
    plt.plot(range(1, num_epochs + 1), rmse_values, label='RMSE', color='red')
    plt.title('RMSE over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')

    plt.tight_layout()
    plt.show()


def log_message(message, log_filepath):
    print(message)
    with open(log_filepath, 'a') as f:
        print(message, file=f)


def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    metrics = {"loss": [], "CI": [], "R2": [], "RMSE": []}
    all_affinity = np.array([])
    y_pred = np.array([])

    with torch.no_grad():
        for batch_i, (XD, XP, y) in enumerate(dataloader):
            drug_encode, protein_encode, affinity = XD.to(
                device), XP.to(device), y.to(device)
            predictions = model(drug_encode, protein_encode)
            y_pred = np.append(
                y_pred, predictions.squeeze().detach().cpu().numpy(), axis=0)
            all_affinity = np.append(all_affinity, affinity.cpu().numpy())

            loss = criterion(predictions.squeeze(), affinity).item()
            ci = cindex_score(affinity.cpu().numpy(),
                              predictions.squeeze().detach().cpu().numpy())
            r2 = r2_score(all_affinity, y_pred)
            rmse = math.sqrt(mean_squared_error(all_affinity, y_pred))

            metrics["loss"].append(loss)
            metrics["CI"].append(ci)
            metrics["R2"].append(r2)
            metrics["RMSE"].append(rmse)

    return {k: round(np.mean(v), 4) for k, v in metrics.items()}


def train_deep_dta(model, train_dataloader, test_dataloader, num_epochs, learning_rate, checkpoint_model='',  early_stopping=125):
    log_filepath = './model_checkpoints/log.txt'
    os.makedirs('./model_checkpoints', exist_ok=True)

    device = DEVICE
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_ci = 0
    epochs_since_improvement = 0

    # Lists for plotting metrics
    loss_values, ci_values, r2_values, rmse_values = [], [], [], []

    # Load checkpoints if available
    if checkpoint_model:
        model.load_state_dict(torch.load(
            checkpoint_model, map_location=device))
        log_message("Model weights loaded.", log_filepath)

    # if checkpoint_optimizer:
    #     optimizer.load_state_dict(torch.load(checkpoint_optimizer, map_location=device))
    #     log_message("Optimizer weights loaded.", log_filepath)

    for epoch in range(num_epochs):
        model.train()
        start_time = time.time()
        train_metrics = {"loss": [], "CI": [], "R2": [], "RMSE": []}
        all_affinity, yl_pred = np.array([]), np.array([])

        for batch_i, (XD, XP, y) in enumerate(train_dataloader):
            drug_encode, protein_encode, affinity = XD.to(
                device), XP.to(device), y.to(device)
            optimizer.zero_grad()
            predictions = model(drug_encode, protein_encode)

            loss = criterion(predictions.squeeze(), affinity)
            ci = cindex_score(affinity.cpu().numpy(),
                              predictions.squeeze().detach().cpu().numpy())

            loss.backward()
            optimizer.step()

            yl_pred = np.append(
                yl_pred, predictions.squeeze().detach().cpu().numpy(), axis=0)
            all_affinity = np.append(all_affinity, affinity.cpu().numpy())

            r2 = r2_score(all_affinity, yl_pred)
            rmse = math.sqrt(mean_squared_error(all_affinity, yl_pred))

            train_metrics["loss"].append(loss.item())
            train_metrics["CI"].append(ci)
            train_metrics["R2"].append(r2)
            train_metrics["RMSE"].append(rmse)

        log_message(
            f"\nEpoch {epoch + 1}/{num_epochs}\n{'=' * 50}", log_filepath)
        train_duration = time.time() - start_time
        log_message(
            f"\nTraining completed in {int(train_duration)}s", log_filepath)

        # Validation
        log_message("Starting validation...", log_filepath)
        val_metrics = evaluate_model(model, test_dataloader, criterion, device)

        log_message(
            f"Validation Metrics: Loss: {val_metrics['loss']}, CI: {val_metrics['CI']}, R2: {val_metrics['R2']}, RMSE: {val_metrics['RMSE']}", log_filepath)

        # Append to metrics lists for plotting
        loss_values.append(val_metrics['loss'])
        ci_values.append(val_metrics['CI'])
        r2_values.append(val_metrics['R2'])
        rmse_values.append(val_metrics['RMSE'])

        if val_metrics['CI'] > best_ci:
            best_ci = val_metrics['CI']
            torch.save(model.state_dict(),
                       './model_checkpoints/best_model.pth')
            torch.save(optimizer.state_dict(),
                       './model_checkpoints/best_optimizer.pth')
            log_message(
                f"Model improved. Saved with CI: {best_ci}", log_filepath)
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        if epochs_since_improvement >= early_stopping:
            log_message("Early stopping triggered.", log_filepath)
            break

    # Plot metrics at the end of training
    plot_fig(loss_values, ci_values, r2_values, rmse_values, len(loss_values))
    return model


if __name__ == "__main__":
    from config import *
    from dataset import DatasetFromCSV
    from torch.utils.data import DataLoader

    # Load data and create DataLoaders
    train_data, valid_data, _ = load_data(CSV_PATH)
    train_dataset = DatasetFromCSV(
        train_data, seqlen=SEQUENCE_LENGTH, smilen=SMILES_LENGTH)
    valid_dataset = DatasetFromCSV(
        valid_data, seqlen=SEQUENCE_LENGTH, smilen=SMILES_LENGTH)

    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Initialize model
    model = DeepDTA().to(DEVICE)

    # Train the model
    model = train_deep_dta(model, train_dataloader, valid_dataloader,
                           num_epochs, learning_rate, checkpoint_model, checkpoint_optimizer)
