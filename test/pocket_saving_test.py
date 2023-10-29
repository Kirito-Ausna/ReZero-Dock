import sys
sys.path.append('/root/Generative-Models/ReDock/')
import pandas as pd
from utils.inference_utils import set_nones
from tqdm import tqdm
from datasets.process_mols import parse_pdb_from_path, read_molecule, rec_atom_featurizer
import numpy as np
import torch
from scipy import spatial
from utils.rotamer import atom_name_vocab
import copy

df = pd.read_csv('/root/Generative-Models/ReDock/data/testset_csv.csv')
complex_name_list = set_nones(df['complex_name'].tolist())
protein_path_list = set_nones(df['protein_path'].tolist())
protein_sequence_list = set_nones(df['protein_sequence'].tolist())
ligand_description_list = set_nones(df['ligand_description'].tolist())

def extract_receptor_structure(rec, lig, lm_embedding_chains=None, pocket_cutoff=8):
    conf = lig.GetConformer()
    lig_coords = conf.GetPositions()
    min_distances = []
    coords = []
    c_alpha_coords = []
    n_coords = []
    c_coords = []
    valid_chain_ids = []
    lengths = []
    pockect_res_masks = [] # pockect residue mask for every chain, including non-valid chain for matching orignal codes
    for i, chain in enumerate(rec):
        chain_coords = []  # num_residues, num_atoms, 3
        chain_c_alpha_coords = []
        chain_n_coords = []
        chain_c_coords = []
        count = 0
        invalid_res_ids = []
        pck_res_mask = []
        for res_idx, residue in enumerate(chain):
            if residue.get_resname() == 'HOH':
                invalid_res_ids.append(residue.get_id())
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
                residue_coords.append(list(atom.get_vector()))
            residue_coords = np.array(residue_coords)
            dist = spatial.distance.cdist(lig_coords, residue_coords).min()
            if c_alpha != None and n != None and c != None and dist <= pocket_cutoff:
            # if c_alpha != None and n != None and c != None:
                # only append residue if it is an amino acid and not some weird molecule that is part of the complex
                chain_c_alpha_coords.append(c_alpha)
                chain_n_coords.append(n)
                chain_c_coords.append(c)
                chain_coords.append(residue_coords)
                pck_res_mask.append(1)
                count += 1
            else:
                if c_alpha != None and n != None and c != None:
                    pck_res_mask.append(0)
                invalid_res_ids.append(residue.get_id())
        for res_id in invalid_res_ids:
            chain.detach_child(res_id)
        
        if len(chain_coords) > 0:
            all_chain_coords = np.concatenate(chain_coords, axis=0)
            distances = spatial.distance.cdist(lig_coords, all_chain_coords)
            min_distance = distances.min()
        else:
            min_distance = np.inf

        min_distances.append(min_distance)
        lengths.append(count)
        coords.append(chain_coords)
        c_alpha_coords.append(np.array(chain_c_alpha_coords))
        n_coords.append(np.array(chain_n_coords))
        c_coords.append(np.array(chain_c_coords))
        pockect_res_masks.append(np.array(pck_res_mask))
        if not count == 0: valid_chain_ids.append(chain.get_id())
    min_distances = np.array(min_distances)
    if len(valid_chain_ids) == 0: # As in current data, this is not possible, but just in case
        valid_chain_ids.append(np.argmin(min_distances))
    valid_coords = []
    valid_c_alpha_coords = []
    valid_n_coords = []
    valid_c_coords = []
    valid_lengths = []
    invalid_chain_ids = []
    valid_lm_embeddings = []
    for i, chain in enumerate(rec):
        if chain.get_id() in valid_chain_ids:
            valid_coords.append(coords[i])
            valid_c_alpha_coords.append(c_alpha_coords[i])
            if lm_embedding_chains is not None:
                if i >= len(lm_embedding_chains):
                    raise ValueError('Encountered valid chain id that was not present in the LM embeddings')
                mask = torch.from_numpy(pockect_res_masks[i]).bool()
                valid_lm_embeddings.append(lm_embedding_chains[i][mask])
                # valid_lm_embeddings.append(lm_embedding_chains[i])
                # if len(lm_embedding_chains[i]) != len(pockect_res_masks[i]):
                    # pdb.set_trace()
            valid_n_coords.append(n_coords[i])
            valid_c_coords.append(c_coords[i])
            valid_lengths.append(lengths[i])
        else:
            invalid_chain_ids.append(chain.get_id())
            
    coords = [item for sublist in valid_coords for item in sublist]  # list with n_residues arrays whose shape in [n_atoms, 3]
    c_alpha_coords = np.concatenate(valid_c_alpha_coords, axis=0)  # [n_residues, 3]
    n_coords = np.concatenate(valid_n_coords, axis=0)  # [n_residues, 3]
    c_coords = np.concatenate(valid_c_coords, axis=0)  # [n_residues, 3]
    lm_embeddings = np.concatenate(valid_lm_embeddings, axis=0) if lm_embedding_chains is not None else None
    for invalid_id in invalid_chain_ids:
        rec.detach_child(invalid_id)

    assert len(c_alpha_coords) == len(n_coords)
    assert len(c_alpha_coords) == len(c_coords)
    assert sum(valid_lengths) == len(c_alpha_coords)
    return rec, coords, c_alpha_coords, n_coords, c_coords, pockect_res_masks

def to_pdb(self, out_path):
        # mimic process_mols.extract_receptor_structure, 
        # but change the coordinates of atoms while parsing in the same order of the preprocess. Just need two loops.
        conf = self.mol.GetConformer()
        lig_coords = conf.GetPositions()
        sc_atom_idx = 0
        for i, chain in enumerate(rec):
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
                    residue_coords.append(list(atom.get_vector()))
                residue_coords = np.array(residue_coords)
                dist = spatial.distance.cdist(lig_coords, residue_coords).min()
                if c_alpha != None and n != None and c != None and dist <= self.pocket_cutoff:
                    # change the coordinates of atoms in the residue
                    for atom in residue:
                        if atom.name not in atom_name_vocab:
                            continue
                        # atoms that should be modified are modified, unchanged keep unchanged
                        # missing keep missing
                        atom.set_coord(self.pocket_pos[sc_atom_idx])
                        sc_atom_idx += 1
        # pdb.set_trace()
        assert sc_atom_idx == len(self.pocket_pos), 'Not all sidechain atoms are modified, index may be mismatched.'

for i, complex_name in tqdm(enumerate(complex_name_list), total=len(complex_name_list)):
    mol = read_molecule(ligand_description_list[i])
    prot = parse_pdb_from_path(protein_path_list[i])
    prot_copy = copy.deepcopy(prot) 
    rec, coords, c_alpha_coords, n_coords, c_coords, pockect_res_masks = extract_receptor_structure(prot_copy, mol)
    # silumate the process of atom_update
    # atom_coords = torch.from_numpy(np.concatenate(coords, axis=0)).float() + 1.257
    atom_coords = torch.from_numpy(np.concatenate(coords, axis=0)).float()
    atom_feat = torch.from_numpy(np.asarray(rec_atom_featurizer(rec)))
    not_hs = (atom_feat[:, 3] < 37)
    atom_coords = atom_coords[not_hs].numpy()
    # silumate the process of pocket saving
    sc_atom_idx = 0
    updated_atoms = {}
    for atom in rec.get_atoms():
        if atom.name not in atom_name_vocab:
            continue
        # atom.set_coord(atom_coords[sc_atom_idx])
        updated_atoms[atom] = atom_coords[sc_atom_idx]
        sc_atom_idx += 1

    assert len(updated_atoms.keys()) == len(atom_coords), 'Not all sidechain atoms are updated, index may be mismatched.'
    modified_atom_num = 0
    for atom in prot.get_atoms():
        if atom in updated_atoms:
            # if (atom.coord - updated_atoms[atom] + 1.257 != 0.0).any():
            #     print('atom {} is modified from {} to {}'.format(atom.name, atom.coord, updated_atoms[atom]))
            atom.set_coord(updated_atoms[atom])
            modified_atom_num += 1
    assert modified_atom_num == len(updated_atoms.keys()), 'Not all sidechain atoms are modified, index may be mismatched.'




