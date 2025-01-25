import numpy as np

def label_smiles(line, MAX_SMI_LEN, smi_ch_ind):
    X = np.zeros(MAX_SMI_LEN)
    for i, ch in enumerate(line[:MAX_SMI_LEN]):
        X[i] = smi_ch_ind.get(ch, 0)  # Default to 0 if character not in dictionary
    return X

def label_sequence(line, MAX_SEQ_LEN, seq_ch_ind):
    X = np.zeros(MAX_SEQ_LEN)
    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i] = seq_ch_ind.get(ch, 0)  # Default to 0 if character not in dictionary
    return X