import os
import pandas as pd

# Directories
ligand_dir = "data/IL-17-data/9FL3/SYNC_9FL3"
protein_dir = "data/IL-17-data/Target_9FL3"
# Save to CSV
output_file = "data/IL-17-data/CSVs/9FL3-SYNC.csv"

# Get the list of molecule files in the ligand directory
ligand_files = [f for f in os.listdir(ligand_dir) if f.endswith('.sdf')]

# Get the protein name (assumed there's only one protein file)
protein_files = [f for f in os.listdir(protein_dir) if f.endswith('.pdb')]

# Check if exactly one protein file is found
if len(protein_files) != 1:
    raise ValueError("There should be exactly one protein file in the Target directory.")

protein_name = protein_files[0].replace('.pdb', '')  # Get the protein name without the extension
protein_path = os.path.join(protein_dir, f"{protein_name}.pdb")

# Initialize list to store data
data = []

# Loop through the ligand files and create corresponding rows
for ligand_file in ligand_files:
    molecule_name = ligand_file.replace('.sdf', '')  # Remove the .sdf extension
    ligand_path = os.path.join(ligand_dir, ligand_file)
    complex_name = molecule_name  # Complex name is the molecule name
    
    # Append data to the list
    data.append([complex_name, ligand_path, protein_path])

# add additional rows with target protein and its crystal ligand for pocket detection
# data.append([protein_name, "data/IL-17-data/Target_8DYG/8DYG_LIG.sdf", protein_path])

# Create a DataFrame
df = pd.DataFrame(data, columns=["complex_name", "ligand_description", "protein_path"])
# add an empty column for protein_sequence
df["protein_sequence"] = ""

df.to_csv(output_file, index=False)

print(f"CSV file generated: {output_file}")