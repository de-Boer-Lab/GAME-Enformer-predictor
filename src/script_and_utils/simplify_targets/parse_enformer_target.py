# parse_enformer_target.py
import os
import pandas as pd

utils_dir = os.path.dirname(__file__)

input_data = pd.read_csv(f'{utils_dir}/enformer_targets_human.txt', sep='\t', header=0)
input_data = input_data[['description']]
print(input_data)

CAGE = input_data[input_data['description'].str.contains("CAGE")]
# Split on the first colon only
CAGE_split = CAGE['description'].str.split(":", n=1, expand=True)
CAGE_split.columns = ['Assay', 'Cell Type']
print(CAGE_split)

DNASE = input_data[input_data['description'].str.startswith(('DNASE', 'ATAC'))]
print(DNASE)
DNASE_split = DNASE['description'].str.split(":", n=1, expand=True)
DNASE_split.columns = ['Assay', 'Cell Type']
print(DNASE_split)

CHIP = input_data[input_data['description'].str.startswith('CHIP')]
CHIP_split = CHIP['description'].str.split(":", n=2, expand=True)
CHIP_split.columns = ['Assay', 'Molecule','Cell Type']

print(CHIP_split)

# Merged data: Note that CHIP will have one extra columns
targets_enformer = pd.concat([DNASE_split, CHIP_split, CAGE_split], ignore_index=True)
targets_enformer.to_csv(f'{utils_dir}/enformer_human_targets_simplified.txt', sep='\t')
