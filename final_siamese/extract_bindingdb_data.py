'''
This script extracts relevant information from the BindingDB dataset:
Parses the molecular data in SDF format.
Extracts molecular properties like SMILES, Molecular Weight, LogP, TPSA, and the Number of Rotatable Bonds.
Saves the extracted data in a format compatible with the merging process.

'''
from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd

# Path to the BindingDB SDF file
bindingdb_path = 'BindingDB_All_2D.sdf'

# Create an SDF supplier with sanitization turned off
suppl = Chem.SDMolSupplier(bindingdb_path, sanitize=False)

# Function to generate molecular descriptors
def generate_descriptors(mol):
    try:
        Chem.SanitizeMol(mol)  # Sanitize molecule to ensure validity
        return {
            'smiles': Chem.MolToSmiles(mol, isomericSmiles=True),
            'MolecularWeight': Descriptors.MolWt(mol),
            'LogP': Descriptors.MolLogP(mol),
            'TPSA': Descriptors.TPSA(mol),
            'NumRotatableBonds': Descriptors.NumRotatableBonds(mol)
        }
    except Exception as e:
        return None

# Batch processing parameters
batch_size = 10000
records = []
batch_number = 1
error_log = []

for i, mol in enumerate(suppl):
    if mol and mol.HasProp("PubChem CID of Ligand"):
        try:
            # Extract PubChem CID
            pubchem_cid = mol.GetProp("PubChem CID of Ligand")
            
            # Extract protein sequence and UniProt ID if available
            protein_sequence = mol.GetProp("BindingDB Target Chain Sequence") if mol.HasProp("BindingDB Target Chain Sequence") else None
            uniprot_id = mol.GetProp("UniProt (SwissProt) Primary ID of Target Chain") if mol.HasProp("UniProt (SwissProt) Primary ID of Target Chain") else None

            # Generate descriptors
            descriptors = generate_descriptors(mol)

            if descriptors:
                record = {
                    'pubchem_cid': pubchem_cid,
                    'smiles': descriptors['smiles'],
                    'MolecularWeight': descriptors['MolecularWeight'],
                    'LogP': descriptors['LogP'],
                    'TPSA': descriptors['TPSA'],
                    'NumRotatableBonds': descriptors['NumRotatableBonds'],
                    'protein_sequence': protein_sequence,
                    'uniprot_id': uniprot_id
                }
                records.append(record)
            else:
                error_log.append(f"Descriptor generation error at molecule {i}")
        except Exception as e:
            error_log.append(f"Error at molecule {i}: {e}")

    # Save batch to CSV every 10,000 records
    if (i + 1) % batch_size == 0:
        batch_df = pd.DataFrame(records)
        batch_filename = f"bindingdb_batch_{batch_number}.csv"
        batch_df.to_csv(batch_filename, index=False)
        print(f"Saved batch {batch_number} with {len(records)} records.")
        records = []
        batch_number += 1

# Save any remaining records
if records:
    batch_df = pd.DataFrame(records)
    batch_filename = f"bindingdb_batch_{batch_number}.csv"
    batch_df.to_csv(batch_filename, index=False)
    print(f"Saved final batch {batch_number} with {len(records)} records.")

# Save the error log to a file
error_log_file = "bindingdb_error_log.txt"
with open(error_log_file, "w") as f:
    for error in error_log:
        f.write(error + "\n")

print(f"Processing complete. Errors logged in '{error_log_file}'")
