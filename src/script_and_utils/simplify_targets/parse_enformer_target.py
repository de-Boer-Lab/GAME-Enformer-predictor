# parse_enformer_target.py
import os
import pandas as pd

utils_dir = os.path.dirname(__file__)

input_data = pd.read_csv(f'{utils_dir}/enformer_targets_mouse.txt', sep='\t', header=0)
input_data = input_data[['description']]
print(input_data)

CAGE = input_data[input_data['description'].str.contains("CAGE")]
# Split on the first colon only
CAGE_split = CAGE['description'].str.split(":", n=1, expand=True)
CAGE_split.columns = ['Assay', 'Cell Type']
print(CAGE_split)

DNASE_ATAC = input_data[input_data['description'].str.startswith(('DNASE', 'ATAC'))]
print(DNASE_ATAC)
DNASE_ATAC_split = DNASE_ATAC['description'].str.split(":", n=1, expand=True)
DNASE_ATAC_split.columns = ['Assay', 'Cell Type']
print(DNASE_ATAC_split)

CHIP = input_data[input_data['description'].str.startswith('CHIP')]
CHIP_split = CHIP['description'].str.split(":", n=2, expand=True)
CHIP_split.columns = ['Assay', 'Molecule','Cell Type']

print(CHIP_split)

# --- THE SAFETY LOCK ---
# 1. Concat the dataframes but KEEP their original indices from input_data
targets_enformer = pd.concat([DNASE_ATAC_split, CHIP_split, CAGE_split])

# 2. Sort by that original index to absolutely guarantee the original file's row order
targets_enformer = targets_enformer.sort_index()

# 3. Now reset the index so it perfectly counts from 0 to 1642 for the mouse tensor
targets_enformer = targets_enformer.reset_index(drop=True)

# Save the simplified file
targets_enformer.to_csv(f'{utils_dir}/enformer_mouse_targets_simplified.txt', sep='\t')
print("\nSuccessfully saved enformer_mouse_targets_simplified.txt")
