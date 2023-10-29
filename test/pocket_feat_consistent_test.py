import pickle
from tqdm import tqdm
# from utils.rotamer import residue_list

residue_list = [
    "GLY", "ALA", "SER", "PRO", "VAL", "THR", "CYS", "ILE", "LEU", "ASN",
    "ASP", "GLN", "LYS", "GLU", "MET", "HIS", "PHE", "ARG", "TYR", "TRP"
]
possible_amino_acids = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET',
                             'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'HIP', 'HIE', 'TPO', 'HID', 'LEV', 'MEU',
                             'PTR', 'GLV', 'CYT', 'SEP', 'HIZ', 'CYM', 'GLM', 'ASQ', 'TYS', 'CYX', 'GLZ', 'misc']

complex_graphs = pickle.load(open('/root/Generative-Models/DiffDock/data/dataset_cache_torsion_allatoms/limit0_INDEXtimesplit_no_lig_overlap_train_maxLigSizeNone_H0_recRad15.0_recMax24_atomRad5_atomMax8_esmEmbeddings/heterographs.pkl', 'rb'))
rdkit_graphs = pickle.load(open('/root/Generative-Models/DiffDock/data/dataset_cache_torsion_allatoms/limit0_INDEXtimesplit_no_lig_overlap_train_maxLigSizeNone_H0_recRad15.0_recMax24_atomRad5_atomMax8_esmEmbeddings/rdkit_ligands.pkl', 'rb'))
inconsistent = []
inconsistent_num = 0
# for idx, complex_graph in enumerate(complex_graphs):
# add tqdm
for idx, complex_graph in tqdm(enumerate(complex_graphs)):
    name = complex_graph['name']
    protein = complex_graph['sidechain']
    len_res_sc = protein.residue_type.shape[0]
    len_res = complex_graph['receptor'].num_nodes
    # assert len_res_sc == len_res, f"The number of residues in the {name} sidechain graph is not equal to the number of residues in the full graph."
    if len_res_sc != len_res:
        inconsistent.append((name,idx))
        inconsistent_num += 1
        print(f"The number of residues in the {name} sidechain graph is not equal to the number of residues in the full graph.")
    
    len_atom_sc = protein.atom2residue.shape[0]
    len_atom = complex_graph['atom'].num_nodes

    if len_atom_sc != len_atom:
        inconsistent.append((name,idx))
        inconsistent_num += 1
        print(f"The number of atoms in the {name} sidechain graph is not equal to the number of atoms in the full graph.")

print(f"Total number of inconsistent graphs: {inconsistent_num}")
print(f"Inconsistent graphs: {inconsistent}")

# save the inconsistent graphs
with open('inconsistent_graphs.pkl', 'wb') as f:
    pickle.dump(inconsistent, f)

# load the inconsistent graphs
with open('/root/Generative-Models/DiffDock/datasets/inconsistent_graphs.pkl', 'rb') as f:
    inconsistent = pickle.load(f)

for name, idx in inconsistent:
    complex_graph = complex_graphs[idx]
    protein = complex_graph['sidechain']
    len_res_sc = protein.residue_type.shape[0]
    len_res = complex_graph['receptor'].num_nodes
    print(f"{name}: ", len_res_sc, len_res)
    res_sc = [residue_list[type] for type in protein.residue_type]
    res = [possible_amino_acids[int(type)] for type in complex_graph['receptor'].x[:,0]]
    pos = complex_graph['receptor'].pos
    # find the position of all the inconsistent residues
    sc_index, index = 0, 0
    wrong_res = []
    while sc_index < len_res_sc and index < len_res:
        if res_sc[sc_index] != res[index]:
            print(f"{name}: ", sc_index, index)
            # sc_index += 1
            wrong_res.append(index)
            index += 1
        else:
            sc_index += 1
            index += 1
    
    for res_id in wrong_res:
        print(f"{name}: ", res[res_id])
        pos_res = pos[res_id]
        print(pos_res)
    
    # complex_graphs.pop(idx)
    # rdkit_graphs.pop(idx)

    print(f"{name}: ", res_sc)
    print(f"{name}: ", res)
    len_atom_sc = protein.atom2residue.shape[0]
    len_atom = complex_graph['atom'].num_nodes
    print(f"{name}: ", len_atom_sc, len_atom)

ids = [idx for name, idx in inconsistent]
# delete the inconsistent complex graph from the list
complex_graphs = [complex_graphs[idx] for idx in range(len(complex_graphs)) if idx not in ids]
# save the consistent graphs
with open('/root/Generative-Models/DiffDock/data/dataset_cache_torsion_allatoms/limit0_INDEXtimesplit_no_lig_overlap_train_maxLigSizeNone_H0_recRad15.0_recMax24_atomRad5_atomMax8_esmEmbeddings/c_heterographs.pkl', 'wb') as f:
    pickle.dump(complex_graphs, f)

with open('/root/Generative-Models/DiffDock/data/dataset_cache_torsion_allatoms/limit0_INDEXtimesplit_no_lig_overlap_train_maxLigSizeNone_H0_recRad15.0_recMax24_atomRad5_atomMax8_esmEmbeddings/c_rdkit_ligands.pkl', 'wb') as f:
    pickle.dump(rdkit_graphs, f)

