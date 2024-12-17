#  https://huggingface.co/datasets/damlab/uniprot?row=0

import pandas as pd

from Bio import SeqIO
from Bio.SeqUtils import gc_fraction
df = pd.read_csv('merged_dataset.csv')
input_file = "uniprot_sprot.fasta"
uni  =[]
seq = []
for seq_record in SeqIO.parse(input_file, "fasta"):
    s = seq_record.id
    s= s.split('|')
    uni.append(s[1])
    seq.append(f'{seq_record.seq}')
dff1 = pd.DataFrame({'UniProt_ID':uni,'seq':seq})
merged_df = df.merge(dff1, on='UniProt_ID', how='left')
merged_df.to_csv('seq_merged_dataset.csv', encoding='utf-8', index=False)