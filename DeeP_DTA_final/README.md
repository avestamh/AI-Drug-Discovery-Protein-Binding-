##  Project Overview

          deep learning model for predicting drug-target affinity

## Summary of Code

   `CIndex Metric (cindex_score.py):`
        Implements Concordance Index (CI), a key performance metric for ranking predictions.

   `Configuration File (config.py):`
        Sets training parameters (e.g., batch size, number of epochs, sequence lengths).
        Defines SMILES and protein character dictionaries.
        Specifies hardware resources (CUDA or CPU).

   `Dataset Loader (dataset.py):`
        Reads CSV data and applies preprocessing:
            SMILES and protein sequences are tokenized into tensors using label_smiles and label_sequence.
            Transforms KIBA scores using -log10(KIBA / 1e9).
       
   ‍`Model (model.py):`
        Implements DeepDTA, which includes:
            Encoders for drug/protein sequences with CNN layers.

  ****  The embedding used in this code is a learned embedding created using nn.Embedding in PyTorch,     whose weights are randomly initialized and optimized during the training process.****
            A multi-head attention mechanism for modeling interactions.
            A final MLP-based regression module to predict affinity.
    **utils.py**: Utility functions for data preprocessing.
   `label_smiles`: Converts SMILES strings to numerical vectors.
   `label_sequence`: Converts protein sequences to numerical vectors.
   `Training (train.py):`
        Handles the model optimization loop with logging and visualization (plots: Loss, CI, R2, RMSE).

   `Testing/Evaluation (test.py):`
        Loads a saved model to evaluate its performance using metrics: Loss, CI, R2, RMSE.

## Workflow
   `Load & Preprocess Data` (dataset.py → SMILES & Protein to indexed tensors).
   `Train DeepDTA Model:`
      Forward pass: Encodes drug/protein sequences, uses attention, then predicts affinity.
      Optimizer updates weights based on MSE loss.
   `Save Model & Test on Unseen Data:`
      Uses evaluate_model to measure CI, R2, RMSE.
## Usage
1. Ensure Python 3.x is installed along with necessary libraries (PyTorch, NumPy, pandas, etc.).
2. Place your dataset CSV file at the path specified in `config.py`.

3. train :  for train code run   `!python train.py`
    To change any of the parameters, such as the number of epochs, you can do    `!python train.py --num_epochs = 10`.

4. test model :  `!python test.py `
      If you need to test with a new dataset, you can change it and just make the test module as above.

5.  To prepare the data, first preprocess_merge_data.py and then add_seq.py that must first be run separately.
## Suggestions for Improvement
   1. Data Checking: Ensure SMILES and sequences are truncated/padded correctly to prevent dimension mismatches.

   2. Use of Pretrained Embeddings
    Instead of using random vectors for protein sequences and SMILES representations, pretrained models such as ProtBERT for protein sequences and ChemBERTa for SMILES representations can be utilized.
    `Advantages:`
        Faster model training by leveraging features extracted from pretrained models.
        Improved performance, especially for small datasets.
    `Recommended Tools:`
        ProtBERT (for protein sequences): HuggingFace (https://huggingface.co/Rostlab/prot_bert) <br>
        ChemBERTa (for molecular SMILES): HuggingFace https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1





## LICENSE
 
