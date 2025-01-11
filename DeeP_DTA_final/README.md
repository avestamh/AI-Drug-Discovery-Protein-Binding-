# DeepDTA - Predicting Drug-Target Binding Affinity

## Project Overview

This project implements a **Deep Learning model** to predict drug-target binding affinity, a critical task in drug discovery. The model uses **SMILES strings** for drug molecules and **protein sequences** for targets, leveraging **CNNs** and **Multi-Head Attention** to achieve accurate predictions of **KIBA scores**.

---

## Summary of Code
read the DeepDTA_ Predicting Protein-Target Binding Affinity Using Deep Learning.pdf for summary of the work
### Data Processing
1. **`preprocess_merge_data.py`**  
   Merges the Deloitte dataset with BindingDB batches based on `pubchem_cid`.  
   Cleans the data and generates `merged_dataset.csv`.

2. **`add_seq.py`**  
   Adds protein sequences to `merged_dataset.csv` using UniProt IDs.  
   Produces `seq_merged_dataset.csv`.

### Model and Utilities
- **`config.py`**: Configuration file with hyperparameters and paths.
- **`dataset.py`**: Loads and preprocesses the dataset, converting SMILES and protein sequences into tensors.
- **`model.py`**: Defines the **DeepDTA model** architecture.
- **`train.py`**: Handles model training and logs metrics.
- **`test.py`**: Evaluates the trained model on a test set.
- **`utils.py`**: Utility functions for encoding SMILES and protein sequences.
- **`cindex_score.py`**: Implements the **Concordance Index (CI)** metric.

---

## Workflow

### 1. Data Processing

1. **Merge the Datasets**  
   Run `preprocess_merge_data.py` to merge the Deloitte dataset and BindingDB batches:  
   ```bash
   python preprocess_merge_data.py
   ```

2. **Add Protein Sequences**  
   Run `add_seq.py` to add protein sequences using UniProt IDs:  
   ```bash
   python add_seq.py
   ```

3. **Final Processed Data**  
   The output of this step is `seq_merged_dataset.csv`, which will be used for training the model.

---

### 2. Model Training and Evaluation

1. **Update `config.py`**  
   Set the path to the processed dataset (`seq_merged_dataset.csv`) in `config.py`:  
   ```python
   CSV_PATH = "path/to/seq_merged_dataset.csv"
   ```

2. **Train the Model**  
   Run `train.py` to train the DeepDTA model:  
   ```bash
   python train.py --num_epochs 20
   ```

3. **Test the Model**  
   Evaluate the model on a test set:  
   ```bash
   python test.py
   ```

---

## Usage Instructions

1. **Ensure Python 3.x** is installed along with the necessary libraries:
   ```bash
   pip install torch numpy pandas matplotlib scikit-learn biopython
   ```

2. **Code Execution Order**:
   1. **Data Processing**:
      - `preprocess_merge_data.py`
      - `add_seq.py`
   2. **Model Training**:
      - `train.py`
   3. **Model Testing**:
      - `test.py`

---

## Suggestions for Improvement

1. **Pretrained Embeddings**:
   - Use **ProtBERT** for protein sequences and **ChemBERTa** for SMILES strings to improve performance.

2. **Hyperparameter Optimization**:
   - Fine-tune learning rates, batch sizes, and CNN architecture.

3. **Graph Neural Networks**:
   - Consider using GNNs for richer drug molecule representations.
