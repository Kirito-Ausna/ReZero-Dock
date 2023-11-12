import os
from argparse import FileType, ArgumentParser
import torch
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from tqdm import tqdm
from Bio import SeqIO
import esm

parser = ArgumentParser()
parser.add_argument('--out_folder', type=str, default="data/esmfold_structures")
parser.add_argument('--protein_ligand_csv', type=str, default='data/testset_csv.csv', help='Path to a .csv specifying the input as described in the main README')
parser.add_argument('--protein_path', type=str, default=None, help='Path to a single PDB file. If this is not None then it will be used instead of the --protein_ligand_csv')
args = parser.parse_args()

three_to_one = {'ALA':	'A',
'ARG':	'R',
'ASN':	'N',
'ASP':	'D',
'CYS':	'C',
'GLN':	'Q',
'GLU':	'E',
'GLY':	'G',
'HIS':	'H',
'ILE':	'I',
'LEU':	'L',
'LYS':	'K',
'MET':	'M',
'MSE':  'M', # MSE this is almost the same AA as MET. The sulfur is just replaced by Selen
'PHE':	'F',
'PRO':	'P',
'PYL':	'O',
'SER':	'S',
'SEC':	'U',
'THR':	'T',
'TRP':	'W',
'TYR':	'Y',
'VAL':	'V',
'ASX':	'B',
'GLX':	'Z',
'XAA':	'X',
'XLE':	'J'}

def get_sequences_from_pdbfile(file_path):
    biopython_parser = PDBParser()
    structure = biopython_parser.get_structure('random_id', file_path)
    structure = structure[0]
    sequence = None
    for i, chain in enumerate(structure):
        seq = ''
        for res_idx, residue in enumerate(chain):
            if residue.get_resname() == 'HOH':
                continue
            residue_coords = []
            c_alpha, n, c = None, None, None
            for atom in residue:
                if atom.name == 'CA':
                    c_alpha = list(atom.get_vector())
                if atom.name == 'N':
                    n = list(atom.get_vector())
                if atom.name == 'C':
                    c = list(atom.get_vector())
            if c_alpha != None and n != None and c != None:  # only append residue if it is an amino acid
                try:
                    seq += three_to_one[residue.get_resname()]
                except Exception as e:
                    seq += '-'
                    print("encountered unknown AA: ", residue.get_resname(), ' in the complex. Replacing it with a dash - .')

        if sequence is None:
            sequence = seq
        else:
            sequence += (":" + seq)

    return sequence

def generate_ESM_structure(model, filename, sequence):
    model.set_chunk_size(256)
    chunk_size = 256
    output = None

    while output is None:
        try:
            with torch.no_grad():
                output = model.infer_pdb(sequence)

            with open(filename, "w") as f:
                f.write(output)
                # print("saved", filename)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory on chunk_size', chunk_size)
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                chunk_size = chunk_size // 2
                if chunk_size > 2:
                    model.set_chunk_size(chunk_size)
                else:
                    print("Not enough memory for ESMFold")
                    break
            else:
                raise e
    return output is not None

if args.protein_path is not None:
    file_paths = [args.protein_path]
else:
    df = pd.read_csv(args.protein_ligand_csv)
    file_paths = list(set(df['protein_path'].tolist()))

model = esm.pretrained.esmfold_v1()
model = model.eval().cuda()

for file_path in tqdm(file_paths, desc="Generating ESMFold structures"):
    sequence = get_sequences_from_pdbfile(file_path)
    pdb_id = os.path.split(os.path.dirname(file_path))[1]
    write_path = os.path.join(args.out_folder, f"{pdb_id}_protein_esmfold.pdb")
    if not os.path.exists(write_path):
        generate_ESM_structure(model, write_path, sequence)
    
    