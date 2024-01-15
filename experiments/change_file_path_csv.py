import sys
sys.path.append('/huangyufei/Generative_Power/ReDock')
import os
import argparse
import pandas as pd
from utils.inference_utils import set_nones
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--protein_ligand_csv_folder", type=str, default="data/crossdock_csv_rep", help="Path to a folder of .csv file specifying the input as described in the README.")
args = parser.parse_args()

for csv_file in tqdm(os.listdir(args.protein_ligand_csv_folder)):
    # delete the str "/home/haotian/Molecule_Generation/Flexible-docking/" in the csv file
    csv_file_path = os.path.join(args.protein_ligand_csv_folder, csv_file)
    df = pd.read_csv(csv_file_path)
    # delete unamed column
    # df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    # delete column "ligand_path", save_path, and target_name
    # df.drop(['ligand_path','save_path', 'target_name'], axis=1, inplace=True)
    # rename the column "ligand_path" to "ligand_description"
    df.rename(columns={'crystal_path': 'ligand_description'}, inplace=True)
    df["protein_sequence"] = None
    # protein_path_list = set_nones(df["protein_path"].tolist())
    # complex_name_list = set_nones(df["complex_name"].tolist())
    # ligand_path_list = set_nones(df["ligand_description"].tolist())
    # new_protein_path_list = []
    # new_complex_name_list = []
    # new_ligand_path_list = []

    # for protein_path in protein_path_list:
    #     new_protein_path_list.append(protein_path.replace('_added', ''))
    # for complex_name in complex_name_list:
    #     new_complex_name_list.append(complex_name.replace('_added', ''))
    # # for ligand_path in ligand_path_list:
    # #     new_ligand_path_list.append(ligand_path.replace('/home/haotian/Molecule_Generation/Flexible-docking/', ''))

    # df['protein_path'] = new_protein_path_list
    # df['complex_name'] = new_complex_name_list
    # df['ligand_description'] = new_ligand_path_list
    # drop the column "ligand_path"
    # df.drop(['ligand_path'], axis=1, inplace=True)
    # replace the original csv file
    df.to_csv(csv_file_path, index=False)