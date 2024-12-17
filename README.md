# **Protein-Ligand Binding Prediction Project**

This repository contains two distinct machine learning approaches for predicting protein-ligand binding interactions. Each approach has its own folder, codebase, and corresponding report.

## **Folder Structure**
**Note:** I could not upload the saved model files: **use_False_data.pth** and **use_True_data.pth** due to size restriction

### 1. **`final_siamese`**

This folder contains the implementation of a **Siamese Neural Network (SNN)** for binary classification of protein-ligand binding (binding vs non-binding).

#### **Contents:**
- **`main.py`**: Main script for training and evaluating the Siamese model.
- **`data_utils.py`**: Utility functions for data loading, preprocessing, and pair generation.
- **`siamese_model.py`**: The Siamese Neural Network architecture.
- **`preprocess_merge_data.py`**: Preprocessing script to merge datasets and prepare features.
- **`test_inference_true_rows.py`**: Script for running inference on test data.
- **Report**: A detailed report explaining the approach, methodology, results, and future improvements.

### 2. **`DeeP_DTA_final`**

This folder contains the implementation of the **DeepDTA** model for predicting continuous binding affinity scores (KIBA scores) between proteins and ligands.

#### **Contents:**
- **`train.py`**: Script for training the DeepDTA model.
- **`test.py`**: Script for evaluating the trained model.
- **`config.py`**: Configuration file with paths and hyperparameters.
- **`add_seq.py`**: Script for adding protein sequences to the dataset.
- **Report**: A detailed report explaining the DeepDTA model, methodology, results, and potential future enhancements.

### 3. **`final_report`**

This file contains the **comprehensive report** summarizing both the **Siamese Neural Network** and **DeepDTA** approaches. It covers:

- **Project Introduction**
- **Data Preparation and Feature Engineering**
- **Detailed Methodologies for Both Models**
- **Results and Comparisons**
- **Challenges and Future Improvements**

## **Summary**

Each folder contains the necessary code, data preprocessing scripts, and a detailed report explaining the approach, methodology, and results. The **final_report** consolidates both approaches into a comprehensive analysis, providing insights into the advantages of each model and suggestions for future work.

---

This structure ensures clarity, reproducibility, and ease of navigation for anyone reviewing or running the code.
