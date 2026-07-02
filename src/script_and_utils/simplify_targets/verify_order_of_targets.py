import pandas as pd

def verify_target_order(original_file, simplified_file):
    print(f"--- Verifying {simplified_file} ---")
    
    # Load files
    orig_df = pd.read_csv(original_file, sep='\t')
    # Since to_csv was used without index=False, we load the first column as the index
    simp_df = pd.read_csv(simplified_file, sep='\t', index_col=0) 

    # 1. Check lengths
    lengths_match = len(orig_df) == len(simp_df)
    print(f"Row counts match: {lengths_match} ({len(orig_df)} rows)")
    if not lengths_match:
        return

    # 2. Reconstruct the original description string
    def reconstruct(row):
        if str(row['Assay']).upper() == 'CHIP':
            return f"CHIP:{row['Molecule']}:{row['Cell Type']}"
        else:
            return f"{row['Assay']}:{row['Cell Type']}"

    reconstructed = simp_df.apply(reconstruct, axis=1).reset_index(drop=True)
    original_desc = orig_df['description'].reset_index(drop=True)

    # 3. Assert 100% exact match
    exact_match = (original_desc == reconstructed).all()
    print(f"100% Exact Order Match: {exact_match}\n")

    # If it fails, print the first few mismatches so you can see where it shifted
    if not exact_match:
        print("Mismatches found:")
        mismatches = original_desc[original_desc != reconstructed]
        for idx in mismatches.index[:5]:
            print(f"Row {idx}:")
            print(f"  Original : {original_desc[idx]}")
            print(f"  Reconstr : {reconstructed[idx]}")

# Run for Mouse
verify_target_order(
    'enformer_targets_mouse.txt', 
    'enformer_mouse_targets_simplified.txt'
)

# Run for Human (update filenames to match your exact human paths if needed)
verify_target_order(
    'enformer_targets_human.txt', 
    'enformer_human_targets_simplified.txt'
)