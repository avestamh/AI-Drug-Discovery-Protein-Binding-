
'''
This is the primary script that orchestrates the entire workflow of the project. It includes the following key components:
Data Loading: Loads and preprocesses the dataset.
Pair Generation: Uses functions from data_utils.py to generate positive and negative protein-ligand pairs.
Model Initialization: Initializes the Siamese Neural Network from siamese_model.py.
Training: Trains the model using the contrastive loss function and saves the best-performing model based on validation loss.
Evaluation: Tests the trained model on a held-out test set and evaluates performance using metrics like accuracy, precision, recall, and F1 score.
Plotting: Generates and saves loss curves and confusion matrices.
Saving Model: siamese_model_best.pth and siamese_model_final.pth

'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_utils import load_data, create_pairs, PairDataset
from siamese_model import SiameseNetwork
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.cuda.empty_cache()

# Hyperparameters
NUM_EPOCHS = 20
LEARNING_RATE = 0.0001
BATCH_SIZE = 32
PATIENCE = 10
DELTA = 0.001
MODEL_SAVE_PATH = "trained_model/siamese_model_best.pth"
# MAX_PAIRS = 100000

MAX_PAIRS = 100000

# Contrastive Loss Class
# Add class weighting to penalizes incorrect predictions of the minority class more heavily
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0, weight_pos=1.0, weight_neg=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.weight_pos = weight_pos
        self.weight_neg = weight_neg

    def forward(self, distance, label):
        pos_loss = self.weight_pos * (1 - label) * torch.pow(distance, 2)
        neg_loss = self.weight_neg * label * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        loss = torch.mean(pos_loss + neg_loss)
        return loss
# Train model function
def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs, patience, delta, model_save_path):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    epochs_no_improve = 0

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0

        # Training step
        for inputs1, inputs2, labels in train_loader:
            # print(f"Batch shapes - inputs1: {inputs1.shape}, inputs2: {inputs2.shape}, labels: {labels.shape}")
            inputs1, inputs2, labels = inputs1.to(device), inputs2.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs1, inputs2)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

        # Validation step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_inputs1, val_inputs2, val_labels in val_loader:
                val_inputs1, val_inputs2, val_labels = val_inputs1.to(device), val_inputs2.to(device), val_labels.to(device)
                val_outputs = model(val_inputs1, val_inputs2)
                val_loss += criterion(val_outputs, val_labels).item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"Validation Loss after Epoch {epoch + 1}: {avg_val_loss:.4f}")

        # Early stopping logic
        if avg_val_loss < best_val_loss - delta:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"Best model saved with validation loss: {best_val_loss:.4f}")
        else:
            epochs_no_improve += 1
            print(f"Early stopping patience counter: {epochs_no_improve}/{patience}")

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

        # Step the scheduler to adjust the learning rate
        scheduler.step(avg_val_loss)

        model.train()

    print("Training complete.")
    return train_losses, val_losses

# Test model function with evaluation metrics
def test_model(model, test_loader):
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for test_inputs1, test_inputs2, test_labels in test_loader:
            test_inputs1, test_inputs2, test_labels = test_inputs1.to(device), test_inputs2.to(device), test_labels.to(device)
            outputs = model(test_inputs1, test_inputs2)
            predictions = (outputs < 0.5).float()  # Assuming a threshold of 0.5 for similarity

            all_labels.extend(test_labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    # Calculate evaluation metrics
    # precision = precision_score(all_labels, all_predictions, zero_division=0)
    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)

    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)
    accuracy = accuracy_score(all_labels, all_predictions)
    cm = confusion_matrix(all_labels, all_predictions)

    # Print evaluation metrics
    print("\nTest Evaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Plot confusion matrix
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['No Bind', 'Bind'], yticklabels=['No Bind', 'Bind'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.show()

# Plot loss curves function
def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('siamese_loss_epoch_%d_batchsiz_%d.png' % (NUM_EPOCHS, BATCH_SIZE))
    plt.show()

# Main function to execute the workflow
def main():
    # Load and prepare data
    print("Loading and preparing data...")
    df = load_data("merged_dataset.csv")

    # Generate pairs and labels once
    all_pairs, all_labels = create_pairs(df, max_pairs=MAX_PAIRS)

    # Stratified splitting into train, validation, and test sets
    all_pairs_array = list(zip(all_pairs, all_labels))
    train_data, temp_data = train_test_split(all_pairs_array, test_size=0.3, stratify=all_labels, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, stratify=[label for _, label in temp_data], random_state=42)

    # Unzip the data into pairs and labels
    train_pairs, train_labels = zip(*train_data)
    val_pairs, val_labels = zip(*val_data)
    test_pairs, test_labels = zip(*test_data)

    # Create DataLoaders using PairDataset
    train_loader = DataLoader(PairDataset(train_pairs, train_labels, device), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(PairDataset(val_pairs, val_labels, device), batch_size=BATCH_SIZE)
    test_loader = DataLoader(PairDataset(test_pairs, test_labels, device), batch_size=BATCH_SIZE)

    # Initialize the Siamese Network model
    input_dim = train_pairs[0][0].shape[0]
    print(f"Training input dimension: {input_dim}")  # Should be 5
    model = SiameseNetwork(input_dim).to(device)

    # Define the contrastive loss, optimizer, and scheduler
    criterion = ContrastiveLoss(margin=1.0, weight_pos=1.0, weight_neg=2.5)  # Adjust weights as needed
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)  # L2 regularization
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6, verbose=True)


    # Train the model and collect losses
    print("Starting training...")
    train_losses, val_losses = train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, NUM_EPOCHS, PATIENCE, DELTA, MODEL_SAVE_PATH)

    # Save the final trained model
    torch.save(model.state_dict(), "trained_model/siamese_model_final.pth")
    print("Final model saved as 'siamese_model_final.pth'")

    # Plot the training and validation losses
    plot_losses(train_losses, val_losses)

    # Test the model
    print("Starting testing...")
    test_model(model, test_loader)

if __name__ == "__main__":
    main()
