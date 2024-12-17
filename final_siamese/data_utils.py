'''
This file contains utility functions for data preprocessing and pair generation:
load_data(file_path):
Loads the dataset, filters rows where kiba_score_estimated=False, performs log transformation on the KIBA score, normalizes numerical features, and resets the index.
create_pairs(df, max_pairs, num_neighbors=10):
Generates balanced positive and negative pairs. Negative pairs are created using K-Nearest Neighbors (KNN) to ensure they are realistic.
PairDataset:
A custom PyTorch Dataset class to handle the loading of pairs for training and evaluation, filtering out any pairs with NaN values.
'''

import random
import pandas as pd
import itertools
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors

def load_data(file_path):
    """
    Load and preprocess the dataset, filtering only rows where kiba_score_estimated is False,
    and dropping rows with missing values in critical features.
    """
    dtype_spec = {
        'UniProt_ID': str,
        'pubchem_cid': str,
        'kiba_score': float,
        'kiba_score_estimated': bool,
        'smiles': str,
        'MolecularWeight': float,
        'LogP': float,
        'TPSA': float,
        'NumRotatableBonds': int
    }

    # Load the dataset with specified data types
    df = pd.read_csv(file_path, dtype=dtype_spec)

    # Drop rows with missing values in 'kiba_score_estimated'
    df.dropna(subset=['kiba_score_estimated'], inplace=True)

    # Filter rows where kiba_score_estimated is False
    df = df[df['kiba_score_estimated'] == False].copy()
    print(f"Total rows with kiba_score_estimated=False: {len(df)}")

    # Drop rows with missing values in critical numerical features
    df.dropna(subset=['MolecularWeight', 'LogP', 'TPSA', 'NumRotatableBonds'], inplace=True)

    # Log-transform the KIBA scores
    df['log_kiba_score'] = np.log1p(df['kiba_score'])

    # Drop the original kiba_score column (optional)
    df = df.drop(columns=['kiba_score'])

      # Reset the index to avoid the index being included as a feature
    df.reset_index(drop=True, inplace=True)

    # Normalize numerical features (excluding non-numeric columns)
    scaler = StandardScaler()
    numerical_columns = ['log_kiba_score', 'MolecularWeight', 'LogP', 'TPSA', 'NumRotatableBonds']


    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    return df


def create_pairs(df, max_pairs, num_neighbors=10):
    """
    Create balanced positive and negative protein-ligand pairs.

    Args:
        df (pd.DataFrame): DataFrame containing 'UniProt_ID', 'pubchem_cid', and features.
        max_pairs (int): Maximum number of pairs to generate.

    Returns:
        pairs (list): List of (feature1, feature2) pairs.
        labels (list): List of labels (1 for positive, 0 for negative).
    """
    pairs = []
    labels = []

    # Define the numerical columns explicitly
    numerical_columns = ['log_kiba_score', 'MolecularWeight', 'LogP', 'TPSA', 'NumRotatableBonds']

    # Extract unique protein-ligand pairs and their features
    unique_pairs = df[['UniProt_ID', 'pubchem_cid']].drop_duplicates().values
    features = df[numerical_columns].values

    # Create a mapping of (protein, ligand) to feature vectors
    pair_to_features = {tuple(pair): feature for pair, feature in zip(unique_pairs, features)}

    # 1. Generate Positive Pairs
    positive_pairs = list(pair_to_features.keys())
    num_positive_pairs = min(len(positive_pairs), max_pairs // 2)

    for pair in random.sample(positive_pairs, num_positive_pairs):
        pairs.append((pair_to_features[pair], pair_to_features[pair]))
        labels.append(1)

    # 2. Generate Negative Pairs using KNN
    ligand_features = df[['MolecularWeight', 'LogP', 'TPSA', 'NumRotatableBonds']].values
    knn = NearestNeighbors(n_neighbors=num_neighbors, algorithm='ball_tree').fit(ligand_features)

    proteins = df['UniProt_ID'].unique()
    ligands = df['pubchem_cid'].unique()
    existing_pairs = set(positive_pairs)

    num_negative_pairs = num_positive_pairs
    generated_negatives = 0

    while generated_negatives < num_negative_pairs:
        protein = random.choice(proteins)
        ligand_index = random.randint(0, len(ligands) - 1)
        ligand = ligands[ligand_index]

        if (protein, ligand) not in existing_pairs:
            # Find the nearest neighbors for the selected ligand
            distances, indices = knn.kneighbors([ligand_features[ligand_index]])
            negative_ligand_index = random.choice(indices[0][1:])  # Avoid the ligand itself

            if negative_ligand_index >= len(ligands):
                continue  # Skip if the index is out of bounds

            negative_ligand = ligands[negative_ligand_index]

            if (protein, negative_ligand) not in existing_pairs:
                protein_feature = df[df['UniProt_ID'] == protein][numerical_columns].iloc[0].values
                ligand_feature = df[df['pubchem_cid'] == negative_ligand][numerical_columns].iloc[0].values

                # Ensure both features have the correct shape of 5
                if protein_feature.shape[0] == 5 and ligand_feature.shape[0] == 5:
                    pairs.append((protein_feature, ligand_feature))
                    labels.append(0)
                    generated_negatives += 1

    print(f"Generated {num_positive_pairs} positive pairs and {generated_negatives} negative pairs.")

    return pairs, labels


    # # Resample to balance the dataset
    # positive_samples = [(pair, label) for pair, label in zip(pairs, labels) if label == 1]
    # negative_samples = [(pair, label) for pair, label in zip(pairs, labels) if label == 0]

    # min_class_size = min(len(positive_samples), len(negative_samples))
    # balanced_pairs = positive_samples[:min_class_size] + negative_samples[:min_class_size]
    # random.shuffle(balanced_pairs)

    # final_pairs, final_labels = zip(*balanced_pairs)
    # return final_pairs, final_labels


class PairDataset(Dataset):
    def __init__(self, pairs, labels, device):
        self.device = device

        # Filter out pairs where either feature1 or feature2 contains NaN values
        filtered_pairs = []
        filtered_labels = []
        for pair, label in zip(pairs, labels):
            feature1, feature2 = np.array(pair[0], dtype=np.float32), np.array(pair[1], dtype=np.float32)
            if not (np.isnan(feature1).any() or np.isnan(feature2).any()):
                filtered_pairs.append((feature1, feature2))
                filtered_labels.append(label)

        self.pairs = filtered_pairs
        self.labels = filtered_labels

        print(f"Filtered dataset size: {len(self.pairs)} pairs")

        # Flag to control printing the feature shape only once
        self.print_flag = True

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        feature1, feature2 = self.pairs[idx]
        label = self.labels[idx]

        # Convert to tensors without moving to device
        feature1 = torch.tensor(feature1, dtype=torch.float32)
        feature2 = torch.tensor(feature2, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)

        # Print feature shapes only once
        if self.print_flag:
            print(f"Feature1 shape: {feature1.shape}, Feature2 shape: {feature2.shape}")
            self.print_flag = False

        return feature1, feature2, label
