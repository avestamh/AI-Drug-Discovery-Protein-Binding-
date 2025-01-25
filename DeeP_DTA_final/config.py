''' Contains configuration parameters for training and testing, such as batch size,
 learning rate, number of epochs, and file paths. 
 It also defines character dictionaries for encoding SMILES and protein sequences.'''

import argparse
import torch

parser = argparse.ArgumentParser(description="Configuration parameters")

## chat default to True if you want to run on False rows, set default=True

parser.add_argument('--use_false_data', type=bool, default=True,
                    help='Use only False data')
parser.add_argument('--use_true_data', type=bool, default=False,
                    help='Use only True data')
parser.add_argument('--load_model', type=bool, default=True,
                    help='Load existing model')
parser.add_argument('--num_epochs', type=int, default=12,
                    help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size for training')
parser.add_argument('--sequence_length', type=int,
                    default=1200, help='Max length for protein sequences')
parser.add_argument('--smiles_length', type=int, default=100,
                    help='Max length for SMILES strings')
parser.add_argument('--csv_path', type=str,
                    default="seq_merged_dataset.csv", help='Path to the CSV file')
parser.add_argument('--learning_rate', type=float,
                    default=1e-5, help='Learning rate for optimizer')
parser.add_argument('--checkpoint_model', type=str,
                    default='./model_checkpoints/best_model.pth', help='Path to model checkpoint')
# parser.add_argument('--checkpoint_optimizer', type=str,
#                     default='model_checkpoints/optimizer.pth', help='Path to optimizer checkpoint')

args = parser.parse_args()

use_False_data = args.use_false_data
use_True_data = args.use_true_data
load_model = args.load_model
num_epochs = args.num_epochs
BATCH_SIZE = args.batch_size
SEQUENCE_LENGTH = args.sequence_length
SMILES_LENGTH = args.smiles_length
CSV_PATH = args.csv_path
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
learning_rate = args.learning_rate
checkpoint_model = args.checkpoint_model
# checkpoint_optimizer = args.checkpoint_optimizer

# Dictionaries to map characters in SMILES or protein sequences to indices
CHARPROTSET = {
    "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, "F": 7, "I": 8, "H": 9,
    "K": 10, "M": 11, "L": 12, "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17,
    "R": 18, "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25
}
CHARPROTLEN = 25

CHARISOSMISET = {
    "#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
    "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
    "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
    "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
    "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
    "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
    "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
    "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64
}
CHARISOSMILEN = 64
