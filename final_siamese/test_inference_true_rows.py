'''
This script performs inference on pairs with kiba_score_estimated=True:
Model Loading:
Loads the trained Siamese Neural Network model.
Inference:
Selects pairs from the dataset and computes the binding prediction.
Displays detailed information about the selected pairs and model output.
Result Interpretation:
Provides the model’s distance output and predicts binding/non-binding based on a threshold.
'''

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from siamese_model import SiameseNetwork

# Paths
MODEL_PATH = "trained_model/siamese_model_best.pth"
DATA_PATH = "merged_dataset.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 1: Load and preprocess data, but now allow kiba_score_estimated=True.
df = pd.read_csv(DATA_PATH, dtype={
    'UniProt_ID': str,
    'pubchem_cid': str,
    'kiba_score': float,
    'kiba_score_estimated': bool,
    'smiles': str,
    'MolecularWeight': float,
    'LogP': float,
    'TPSA': float,
    'NumRotatableBonds': int
})

# Filter rows with kiba_score_estimated=True
df = df[df['kiba_score_estimated'] == True].copy()

# Drop rows with missing critical numerical features
df.dropna(subset=['MolecularWeight', 'LogP', 'TPSA', 'NumRotatableBonds'], inplace=True)

# Log-transform the KIBA scores
df['log_kiba_score'] = np.log1p(df['kiba_score'])
df.drop(columns=['kiba_score'], inplace=True)

# Normalize numerical features
numerical_columns = ['log_kiba_score', 'MolecularWeight', 'LogP', 'TPSA', 'NumRotatableBonds']
scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Step 2: Select a single pair. We need two entries:
# Ensure at least two rows are available
if len(df) < 2:
    raise ValueError("Not enough rows with kiba_score_estimated=True to form a pair.")

# Pick two rows (in a real scenario, you might pick a known binding pair or a hypothetical pair)
row1 = df.iloc[0]
row2 = df.iloc[1]

# Print the selected pairs (UniProt IDs and PubChem CIDs)
print(f"Selected Pair 1: UniProt_ID = {row1['UniProt_ID']}, PubChem_CID = {row1['pubchem_cid']}")
print(f"Selected Pair 2: UniProt_ID = {row2['UniProt_ID']}, PubChem_CID = {row2['pubchem_cid']}")

# Check if these pairs exist in the dataset
pair1_exists = ((df['UniProt_ID'] == row1['UniProt_ID']) & (df['pubchem_cid'] == row1['pubchem_cid'])).any()
pair2_exists = ((df['UniProt_ID'] == row2['UniProt_ID']) & (df['pubchem_cid'] == row2['pubchem_cid'])).any()

print(f"Pair 1 exists in the dataset: {'Yes' if pair1_exists else 'No'}")
print(f"Pair 2 exists in the dataset: {'Yes' if pair2_exists else 'No'}")

# Extract feature vectors (excluding categorical cols)
feature_cols = ['log_kiba_score', 'MolecularWeight', 'LogP', 'TPSA', 'NumRotatableBonds']
feature1 = row1[feature_cols].values.astype(np.float32)
feature2 = row2[feature_cols].values.astype(np.float32)

# Print the features for each pair
print(f"Feature Vector 1: {feature1}")
print(f"Feature Vector 2: {feature2}")

feature1_tensor = torch.tensor(feature1, dtype=torch.float32).to(device).unsqueeze(0)  # [1, feature_dim]
feature2_tensor = torch.tensor(feature2, dtype=torch.float32).to(device).unsqueeze(0)

# Step 3: Load the trained model
# Ensure input_dim matches what you had during training
input_dim = len(feature_cols)
model = SiameseNetwork(input_dim).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Step 4: Run inference
with torch.no_grad():
    output = model(feature1_tensor, feature2_tensor)
    # The output is a distance measure. If the distance is less than 0.5, predict binding
    prediction = "Predicted Binding" if output.item() < 0.5 else "Predicted Non-Binding"

    # Print the raw output and the predicted binding status
    print(f"Model output (distance): {output.item():.4f}")
    print(prediction)
