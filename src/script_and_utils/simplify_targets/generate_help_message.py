import json
import pandas as pd

# Load both simplified targets
human_df = pd.read_csv('enformer_human_targets_simplified.txt', sep='\t')
mouse_df = pd.read_csv('enformer_mouse_targets_simplified.txt', sep='\t')

# Create a master dataframe for the help file
master_df = pd.concat([human_df, mouse_df], ignore_index=True)

def feature_label(row):
    if row['Assay'] == 'CHIP' and pd.notna(row['Molecule']):
        return f"CHIP_{row['Molecule']}"
    return row['Assay']


# Generate the JSON structure
help_data = {
    "model": "Enformer",
    "game_schema_version": "1.0",
    "publication": "Avsec et. al, 2021 (Nature Methods)",
    "container_authors": ["Ishika Luthra", "Satyam Priyadarshi"],
    "input_size": 196608,
    "bin_size": 128,
    "expression_strand_specific": False,
    "species": ["homo_sapiens"] * len(human_df) + ["mus_musculus"] * len(mouse_df),
    "features": master_df.apply(feature_label, axis=1).tolist(),
    "cell_types": master_df['Cell Type'].tolist()
}

with open('new_enformer_help_message.json', 'w') as f:
    json.dump(help_data, f, indent=2)