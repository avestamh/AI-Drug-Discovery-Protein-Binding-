import pandas as pd
import glob
import os

# ------------------------------- 1. Load and Preprocess Deloitte Dataset -------------------------------

def preprocess_deloitte_data(filepath):
    """
    Load and preprocess the Deloitte dataset by handling missing values and ensuring data consistency.
    """
    print("Loading Deloitte dataset...")
    df = pd.read_csv(filepath)

    print(f"Original Deloitte dataset size: {len(df)} rows")

    # Drop rows with missing 'pubchem_cid' or 'kiba_score_estimated'
    df.dropna(subset=['pubchem_cid', 'kiba_score_estimated'], inplace=True)

    # Ensure 'pubchem_cid' is of type string for consistency
    df['pubchem_cid'] = df['pubchem_cid'].astype(str)

    print(f"Preprocessed Deloitte dataset size: {len(df)} rows")
    return df

# ------------------------------- 2. Load and Preprocess BindingDB Batches -------------------------------

def preprocess_bindingdb_batches(batch_dir):
    """
    Load and preprocess all BindingDB batch files by concatenating them and handling missing values.
    """
    print("Loading BindingDB batch files...")

    # Use glob to dynamically find all CSV files in the batch directory
    bindingdb_batch_files = glob.glob(os.path.join(batch_dir, "bindingdb_batch_*.csv"))

    # Debug: print the list of batch files found
    print(f"Found {len(bindingdb_batch_files)} batch files.")

    if not bindingdb_batch_files:
        raise FileNotFoundError(f"No batch files found in {batch_dir}. Please check the directory and file names.")

    # Load and concatenate all BindingDB batches
    bindingdb_df = pd.concat([pd.read_csv(f) for f in bindingdb_batch_files], ignore_index=True)

    print(f"Original BindingDB dataset size: {len(bindingdb_df)} rows")

    # Drop rows with missing ligand descriptors
    bindingdb_df.dropna(subset=['pubchem_cid', 'smiles', 'MolecularWeight', 'LogP', 'TPSA', 'NumRotatableBonds'], inplace=True)

    # Ensure 'pubchem_cid' is of type string for consistency
    bindingdb_df['pubchem_cid'] = bindingdb_df['pubchem_cid'].astype(str)

    print(f"Preprocessed BindingDB dataset size: {len(bindingdb_df)} rows")
    return bindingdb_df

# ------------------------------- 3. Merge Datasets -------------------------------

def merge_datasets(deloitte_df, bindingdb_df, output_filepath):
    """
    Merge the Deloitte dataset with the BindingDB dataset on 'pubchem_cid' and handle missing values in the merged data.
    """
    print("Merging datasets on 'pubchem_cid'...")

    # Perform the merge
    merged_df = deloitte_df.merge(bindingdb_df, on='pubchem_cid', how='left')

    print(f"Size after merging: {len(merged_df)} rows")

    # Drop rows with missing ligand descriptors after merging
    merged_df.dropna(subset=['smiles', 'MolecularWeight', 'LogP', 'TPSA', 'NumRotatableBonds'], inplace=True)

    # Drop rows with missing 'kiba_score_estimated'
    merged_df.dropna(subset=['kiba_score_estimated'], inplace=True)

    # Remove duplicates based on 'UniProt_ID' and 'pubchem_cid'
    merged_df.drop_duplicates(subset=['UniProt_ID', 'pubchem_cid'], keep='first', inplace=True)

        # Ensure there are no missing values in the final dataset
    merged_df.dropna(inplace=True)
    print(f"Size after removing duplicates: {len(merged_df)} rows")

    # Save the final merged dataset
    merged_df.to_csv(output_filepath, index=False)
    print(f"Final merged dataset saved to '{output_filepath}'")


# ------------------------------- 4. Main Function -------------------------------

def main():
    # Paths to datasets
    deloitte_filepath = "/home/sadra/Desktop/deloitte_project/Deloitte_DrugDiscovery_dataset.csv"
    batch_dir = "/home/sadra/Desktop/deloitte_project/binding_batches_sequence"
    output_filepath = "merged_dataset_with_seq.csv"

    # Preprocess Deloitte dataset
    deloitte_df = preprocess_deloitte_data(deloitte_filepath)

    # Preprocess BindingDB batches
    bindingdb_df = preprocess_bindingdb_batches(batch_dir)

    # Merge datasets and save the result
    merge_datasets(deloitte_df, bindingdb_df, output_filepath)

if __name__ == "__main__":
    main()
