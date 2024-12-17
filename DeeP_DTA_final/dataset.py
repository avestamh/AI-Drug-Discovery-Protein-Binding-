
'''Loads and preprocesses the dataset, converts SMILES strings and protein sequences into numerical tensors,
 and applies transformations to the KIBA scores. 
 It also defines a custom PyTorch dataset class for loading the data.
 '''

import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from utils import label_smiles, label_sequence
from config import *
import numpy as np

# Custom PyTorch dataset class for loading data from a CSV file

class DatasetFromCSV(Dataset):
    def __init__(self, data, seqlen=SEQUENCE_LENGTH, smilen=SMILES_LENGTH):
        self.SEQLEN = seqlen
        self.SMILEN = smilen
        self.charseqset = CHARPROTSET
        self.charsmiset = CHARISOSMISET
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        smiles = label_smiles(row['smiles'], self.SMILEN, self.charsmiset)
        sequence = label_sequence(row['seq'], self.SEQLEN, self.charseqset)
        affinity = row['kiba_score']
        affinity = -(np.log10(affinity / (10 ** 9)))
        return (torch.tensor(smiles, dtype=torch.long),
                torch.tensor(sequence, dtype=torch.long),
                torch.tensor(affinity, dtype=torch.float32))

# Function to load and split the dataset into training, validation, and test sets

def load_data(csv_path, test_size=0.2):
    data = pd.read_csv(csv_path)
  # Filter the data based on the `kiba_score_estimated` flag

    if use_False_data:
        print('------use False data-----')
        data = data[data['kiba_score_estimated'] == False]
    elif use_True_data:
        print('------use True data-----')
        data = data[data['kiba_score_estimated'] == True]
    else:
        print("use all data")

    train_data, test_data = train_test_split(
        data, test_size=test_size, random_state=0)

    train_data_final, valid_data = train_test_split(
        train_data, test_size=0.05, random_state=0)

    return train_data_final.reset_index(drop=True), valid_data.reset_index(drop=True), test_data.reset_index(drop=True)