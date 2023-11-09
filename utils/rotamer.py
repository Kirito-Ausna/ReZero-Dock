import math
import numpy as np
import copy
import torch
import pdb
from datasets.process_mols import safe_index
from torch_geometric.utils import subgraph

residue_list = [
    "GLY", "ALA", "SER", "PRO", "VAL", "THR", "CYS", "ILE", "LEU", "ASN",
    "ASP", "GLN", "LYS", "GLU", "MET", "HIS", "PHE", "ARG", "TYR", "TRP"
]

residue_list_expand = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET',
                       'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'HIP', 'HIE', 'TPO', 'HID', 'LEV', 'MEU',
                       'PTR', 'GLV', 'CYT', 'SEP', 'HIZ', 'CYM', 'GLM', 'ASQ', 'TYS', 'CYX', 'GLZ', 'misc'],

restype_name_to_atom14_names = {
    "ALA": ["N", "CA", "C", "O", "CB", "", "", "", "", "", "", "", "", ""],
    "ARG": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD",
        "NE",
        "CZ",
        "NH1",
        "NH2",
        "",
        "",
        "",
    ],
    "ASN": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "OD1",
        "ND2",
        "",
        "",
        "",
        "",
        "",
        "",
    ],
    "ASP": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "OD1",
        "OD2",
        "",
        "",
        "",
        "",
        "",
        "",
    ],
    "CYS": ["N", "CA", "C", "O", "CB", "SG", "", "", "", "", "", "", "", ""],
    "GLN": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD",
        "OE1",
        "NE2",
        "",
        "",
        "",
        "",
        "",
    ],
    "GLU": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD",
        "OE1",
        "OE2",
        "",
        "",
        "",
        "",
        "",
    ],
    "GLY": ["N", "CA", "C", "O", "", "", "", "", "", "", "", "", "", ""],
    "HIS": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "ND1",
        "CD2",
        "CE1",
        "NE2",
        "",
        "",
        "",
        "",
    ],
    "ILE": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG1",
        "CG2",
        "CD1",
        "",
        "",
        "",
        "",
        "",
        "",
    ],
    "LEU": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "",
        "",
        "",
        "",
        "",
        "",
    ],
    "LYS": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD",
        "CE",
        "NZ",
        "",
        "",
        "",
        "",
        "",
    ],
    "MET": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "SD",
        "CE",
        "",
        "",
        "",
        "",
        "",
        "",
    ],
    "PHE": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "CE1",
        "CE2",
        "CZ",
        "",
        "",
        "",
    ],
    "PRO": ["N", "CA", "C", "O", "CB", "CG", "CD", "", "", "", "", "", "", ""],
    "SER": ["N", "CA", "C", "O", "CB", "OG", "", "", "", "", "", "", "", ""],
    "THR": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "OG1",
        "CG2",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
    ],
    "TRP": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "NE1",
        "CE2",
        "CE3",
        "CZ2",
        "CZ3",
        "CH2",
    ],
    "TYR": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "CE1",
        "CE2",
        "CZ",
        "OH",
        "",
        "",
    ],
    "VAL": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG1",
        "CG2",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
    ],
    "UNK": ["", "", "", "", "", "", "", "", "", "", "", "", "", ""],
}

atom_name_vocab = {
    "C": 0, "CA": 1, "CB": 2, "CD": 3, "CD1": 4, "CD2": 5, "CE": 6, "CE1": 7, "CE2": 8,
    "CE3": 9, "CG": 10, "CG1": 11, "CG2": 12, "CH2": 13, "CZ": 14, "CZ2": 15, "CZ3": 16,
    "N": 17, "ND1": 18, "ND2": 19, "NE": 20, "NE1": 21, "NE2": 22, "NH1": 23, "NH2": 24,
    "NZ": 25, "O": 26, "OD1": 27, "OD2": 28, "OE1": 29, "OE2": 30, "OG": 31, "OG1": 32,
    "OH": 33, "OXT": 34, "SD": 35, "SG": 36,
}

atom_name2id = {"C": 0, "CA": 1, "CB": 2, "CD": 3, "CD1": 4, "CD2": 5, "CE": 6, "CE1": 7, "CE2": 8,
                    "CE3": 9, "CG": 10, "CG1": 11, "CG2": 12, "CH2": 13, "CZ": 14, "CZ2": 15, "CZ3": 16,
                    "N": 17, "ND1": 18, "ND2": 19, "NE": 20, "NE1": 21, "NE2": 22, "NH1": 23, "NH2": 24,
                    "NZ": 25, "O": 26, "OD1": 27, "OD2": 28, "OE1": 29, "OE2": 30, "OG": 31, "OG1": 32,
                    "OH": 33, "OXT": 34, "SD": 35, "SG": 36, "UNK": 37}

bb_atom_name = [atom_name_vocab[_] for _ in ['C', 'CA', 'N', 'O']]

chi_angles_atoms = {
    'ALA': [],
    # Chi5 in arginine is always 0 +- 5 degrees, so ignore it.
    'ARG': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'],
            ['CB', 'CG', 'CD', 'NE'], ['CG', 'CD', 'NE', 'CZ']],
    'ASN': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'OD1']],
    'ASP': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'OD1']],
    'CYS': [['N', 'CA', 'CB', 'SG']],
    'GLN': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'],
            ['CB', 'CG', 'CD', 'OE1']],
    'GLU': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'],
            ['CB', 'CG', 'CD', 'OE1']],
    'GLY': [],
    'HIS': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'ND1']],
    'ILE': [['N', 'CA', 'CB', 'CG1'], ['CA', 'CB', 'CG1', 'CD1']],
    'LEU': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
    'LYS': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'],
            ['CB', 'CG', 'CD', 'CE'], ['CG', 'CD', 'CE', 'NZ']],
    'MET': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'SD'],
            ['CB', 'CG', 'SD', 'CE']],
    'PHE': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
    'PRO': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD']],
    'SER': [['N', 'CA', 'CB', 'OG']],
    'THR': [['N', 'CA', 'CB', 'OG1']],
    'TRP': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
    'TYR': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
    'VAL': [['N', 'CA', 'CB', 'CG1']],
}

"""
restype_atomname_index_map[i_resi][j_atom]:
  i_resi: index of residue_list, specifies resi_type
  j_atom: atom name, 0-36
  value: index in residue type, 0-13, specifies atom_type, -1 means no atoms
"""
restype_atom14_index_map = -torch.ones((len(residue_list), 37), dtype=torch.long)
for i_resi, resi_name3 in enumerate(residue_list):
    for value, name in enumerate(restype_name_to_atom14_names[resi_name3]):
        if name in atom_name_vocab:
            restype_atom14_index_map[i_resi][atom_name_vocab[name]] = value


"""
chi_atom_index_map[i_resi][j_chi][k_atom]:
  i_resi: index of residue_list, specifies resi_type
  j_chi: chi number, 0-3
  k_atom: k-th atom in the torsion angle, 0-3
  value: index of atom_names, specifies atom_type, -1 means no such torsion
chi_atom14_index_map[i_resi][j_chi][k_atom]:
  value: index in residue type, 0-13, specifies atom_type, -1 means no atoms
"""
chi_atom37_index_map = -torch.ones((len(residue_list), 4, 4), dtype=torch.long)
chi_atom14_index_map = -torch.ones((len(residue_list), 4, 4), dtype=torch.long)
for i_resi, resi_name3 in enumerate(residue_list):
    chi_angles_atoms_i = chi_angles_atoms[resi_name3]
    for j_chi, atoms in enumerate(chi_angles_atoms_i):
        for k_atom, atom in enumerate(atoms):
            chi_atom37_index_map[i_resi][j_chi][k_atom] = atom_name_vocab[atom]
            chi_atom14_index_map[i_resi][j_chi][k_atom] = restype_atom14_index_map[i_resi][atom_name_vocab[atom]]
# Masks out non-existent torsions.
chi_masks = chi_atom37_index_map != -1

# Atom Symmetry
symm_sc_res_atoms = {
    "ARG": [["NH1", "NH2"], ["OXT", "OXT"]],  # ARG *
    "HIS": [["ND1", "CD2"], ["NE2", "CE1"]],  # HIS * *
    "ASP": [["OD1", "OD2"], ["OXT", "OXT"]],  # ASP *
    "PHE": [["CD1", "CD2"], ["CE1", "CE2"]],  # PHE * *
    "GLN": [["OE1", "NE2"], ["OXT", "OXT"]],  # GLN - check
    "GLU": [["OE1", "OE2"], ["OXT", "OXT"]],  # GLU *
    "LEU": [["CD1", "CD2"], ["OXT", "OXT"]],  # LEU - check
    "ASN": [["OD1", "ND2"], ["OXT", "OXT"]],  # ASN - check
    "TYR": [["CD1", "CD2"], ["CE1", "CE2"]],  # TYR * *
    "VAL": [["CG1", "CG2"], ["OXT", "OXT"]],  # VAL - check
}
res_sym_atom_posn = -torch.ones(len(residue_list), 2, 2, dtype=torch.long)
for res, [[a, b], [c, d]] in symm_sc_res_atoms.items():
    res_sym_atom_posn[safe_index(residue_list, res)] = torch.tensor([
        [atom_name_vocab[c], atom_name_vocab[d]],
        [atom_name_vocab[a], atom_name_vocab[b]]
    ])

def get_atom37_position(protein, node_position=None, return_nan=True):
    """
    Get the position of 37 atoms for each residue

    Args:
        protein: Protein object
        node_position: (num_atom, 3)
        return_nan: whether to return nan if the atom is not in the 37 atoms

    Returns:
        node_position37: (num_residue, 37, 3): nan if the atom is not in the 37 atoms
    """
    if node_position is None:
        node_position = protein.node_position
    device = node_position.device
    if return_nan:
        # pdb.set_trace()
        node_position37 = torch.ones((protein.num_nodes, 37, 3), dtype=torch.float, device=device) * np.nan
    else:
        node_position37 = torch.zeros((protein.num_nodes, 37, 3), dtype=torch.float, device=device)
    # pdb.set_trace()
    node_position37[protein.atom2residue, protein.atom_name, :] = node_position
    return node_position37

def get_chi_atom_position(protein, node_position=None):
    """
    Get atom position for each chi torsion angles of each residue.

    Args:
        protein: Protein object.
        node_position: (num_atom, 3) tensor, atom position.

    Returns:
        chi_atom_position: (num_residue, 4, 4, 3) tensor, atom position for each chi torsion angles of each residue.
        `Nan` indicates that the atom does not exist.
    """
    if node_position is None:
        node_position = protein.node_position
    node_position37 = get_atom37_position(protein, node_position)
    # pdb.set_trace()
    chi_atom37_index = chi_atom37_index_map.to(node_position.device)[protein.residue_type]    # (num_residue, 4, 4) 0~36
    chi_atom37_mask = chi_atom37_index == -1
    chi_atom37_index[chi_atom37_mask] = 0
    chi_atom37_index = chi_atom37_index.flatten(-2, -1)                                 # (num_residue, 16)
    chi_atom_position = torch.gather(node_position37, -2,
                                     chi_atom37_index[:, :, None].expand(-1, -1, 3))    # (num_residue, 16, 3)
    chi_atom_position = chi_atom_position.view(-1, 4, 4, 3)                             # (num_residue, 4, 4, 3)
    return chi_atom_position

def get_chi_mask(protein, chi_id=None, device=None):
    # pdb.set_trace()
    chi_atom14_index = chi_atom14_index_map.to(device)[protein.residue_type]  # (num_residue, 4, 4) 0~13
    chi_atom14_mask = chi_atom14_index != -1
    chi_mask = chi_atom14_mask.all(dim=-1)  # (num_residue, 4)

    chi_atom_position = get_chi_atom_position(protein)  # (num_residue, 4, 4, 3)
    has_atom_mask = ~torch.isnan(chi_atom_position).any(dim=-1).any(dim=-1)  # (num_residue, 4)
    # pdb.set_trace()
    chi_mask = chi_mask & has_atom_mask

    if chi_id is not None:
        chi_mask[:, :chi_id] = False
        chi_mask[:, chi_id + 1:] = False
    return chi_mask


# Angle Symmetry
chi_pi_periodic_dict = {
    "ALA": [False, False, False, False],  # ALA
    "ARG": [False, False, False, False],  # ARG
    "ASN": [False, False, False, False],  # ASN
    "ASP": [False, True, False, False],  # ASP
    "CYS": [False, False, False, False],  # CYS
    "GLN": [False, False, False, False],  # GLN
    "GLU": [False, False, True, False],  # GLU
    "GLY": [False, False, False, False],  # GLY
    "HIS": [False, False, False, False],  # HIS
    "ILE": [False, False, False, False],  # ILE
    "LEU": [False, False, False, False],  # LEU
    "LYS": [False, False, False, False],  # LYS
    "MET": [False, False, False, False],  # MET
    "PHE": [False, True, False, False],  # PHE
    "PRO": [False, False, False, False],  # PRO
    "SER": [False, False, False, False],  # SER
    "THR": [False, False, False, False],  # THR
    "TRP": [False, False, False, False],  # TRP
    "TYR": [False, True, False, False],  # TYR
    "VAL": [False, False, False, False],  # VAL
}

chi_pi_periodic = [chi_pi_periodic_dict[res_name] for res_name in residue_list]

def sub_atom_graph(data, atom_mask):
    # pdb.set_trace()
    src_c_alpha_idx = data['sidechain'].atom2residue # H atoms have been removed
    protein = data['sidechain']
    # subgraph
    src_c_alpha_idx = src_c_alpha_idx[atom_mask]
    data['atom'].x = data['atom'].x[atom_mask]
    data['atom'].pos = data['atom'].pos[atom_mask]
    data['atom'].batch = data['atom'].batch[atom_mask] # for batched graph
    # apply atom mask to tensor dict data['atom'].node_t
    for key in data['atom'].node_t.keys():
        data['atom'].node_t[key] = data['atom'].node_t[key][atom_mask]
    src_atom_idx = torch.arange(len(data['atom'].x), device=src_c_alpha_idx.device)
    # atom_res_edge_index = torch.concat((src_atom_idx, src_c_alpha_idx), dim=0).long()
    # concat src_atom_idx and src_c_alpha_idx to [2, num_edge]
    # pdb.set_trace()
    atom_res_edge_index = torch.stack((src_atom_idx, src_c_alpha_idx), dim=0).long()
    # pdb.set_trace()
    data['atom', 'atom_rec_contact', 'receptor'].edge_index = atom_res_edge_index
    # Update the atom_atom edge_index accorddig to atom mask
    # pdb.set_trace()
    atom_edge_index = data['atom', 'atom_contact', 'atom'].edge_index
    atom_edge_index, _ = subgraph(atom_mask, atom_edge_index, relabel_nodes=True)
    data['atom', 'atom_contact', 'atom'].edge_index = atom_edge_index
    # data['atom'].src_c_alpha_idx = torch.from_numpy(src_c_alpha_idx)
    # protein = data['sidechain']
    protein.node_position = data['atom'].pos
    protein.atom_name = protein.atom_name[atom_mask]
    protein.atom2residue = src_c_alpha_idx
    protein.atom14index = protein.atom14index[atom_mask] 
    
    return data

@torch.no_grad()
def remove_by_chi(data, chi_id):
    # clone the protein dict
    #NOTE: clone() is necessary, otherwise the original protein will be modified
    # we clone the heterograph in get_item() method
    new_protein = data['sidechain']
    mask_attrs = ['chi_1pi_periodic_mask', 'chi_2pi_periodic_mask', 'chi_mask']
    # pdb.set_trace()
    for attr in mask_attrs:
        if hasattr(new_protein, attr):
            getattr(new_protein, attr)[:, :chi_id] = 0
            getattr(new_protein, attr)[:, chi_id + 1:] = 0
    if chi_id == 3:
        return data
    else:
        chi_atom14_index = chi_atom14_index_map.to(new_protein.node_position.device)[new_protein.residue_type]
        atom_4 = chi_atom14_index[:, chi_id + 1, -1]
        atom_mask = (new_protein.atom14index >= atom_4[new_protein.atom2residue]) & (
                atom_4[new_protein.atom2residue] != -1)
        # new_protein = new_protein.subgraph(~atom_mask)
        # old_data = data
        data = sub_atom_graph(data, ~atom_mask)
        
        return data

def get_dihedral(p0, p1, p2, p3):
    """
    Given p0-p3, compute dihedral b/t planes p0p1p2 and p1p2p3.
    """
    assert p0.shape[-1] == p1.shape[-1] == p2.shape[-1] == p3.shape[-1] == 3

    # dx
    b0 = p1 - p0
    b1 = p2 - p1
    b2 = p3 - p2

    # normals
    n012 = torch.cross(b0, b1)
    n123 = torch.cross(b1, b2)

    # dihedral
    cos_theta = torch.einsum('...i,...i->...', n012, n123) / (
            torch.norm(n012, dim=-1) * torch.norm(n123, dim=-1) + 1e-10)
    sin_theta = torch.einsum('...i,...i->...', torch.cross(n012, n123), b1) / (
            torch.norm(n012, dim=-1) * torch.norm(n123, dim=-1) * torch.norm(b1, dim=-1) + 1e-10)
    theta = torch.atan2(sin_theta, cos_theta)

    return theta

def get_chis(protein, node_position=None):
    if node_position is None:
        node_position = protein.node_position
    # pdb.set_trace()
    chi_atom_position = get_chi_atom_position(protein, node_position)   # (num_residue, 4, 4, 3)
    chis = get_dihedral(*chi_atom_position.unbind(-2))                  # (num_residue, 4)
    chis[~protein.chi_mask] = np.nan
    return chis

def get_atom14_position(protein, node_position=None):
    """
    Get the position of 14 atoms for each residue
    Args:
        protein: Protein object
        node_position: (num_atom, 3)

    Returns:
        node_position14: (num_residue, 14, 3)
        mask14: (num_atom,) indicate whether the atom is in the 14 atoms
    """
    if node_position is None:
        node_position = protein.node_position
    atom14index = restype_atom14_index_map.to(node_position.device)[
        protein.residue_type[protein.atom2residue], protein.atom_name
    ]  # (num_atom, )
    node_position14 = torch.zeros((protein.num_nodes, 14, 3), dtype=torch.float, device=node_position.device)
    mask14 = atom14index != -1  # (num_atom, )
    node_position14[protein.atom2residue[mask14], atom14index[mask14], :] = node_position[mask14]
    return node_position14, mask14

@torch.no_grad()
def rotate_side_chain(protein, rotate_angles):
    assert rotate_angles.shape[0] == protein.num_nodes
    assert rotate_angles.shape[1] == 4
    node_position14, mask14 = get_atom14_position(protein)  # (num_residue, 14, 3)

    chi_atom14_index = chi_atom14_index_map.to(mask14.device)[protein.residue_type]  # (num_residue, 4, 4) 0~13
    chi_atom14_mask = chi_atom14_index != -1
    chi_atom14_index[~chi_atom14_mask] = 0
    for i in range(4):
        atom_1, atom_2, atom_3, atom_4 = chi_atom14_index[:, i, :].unbind(-1)  # (num_residue, )
        atom_2_position = torch.gather(node_position14, -2,
                                       atom_2[:, None, None].expand(-1, -1, 3))  # (num_residue, 1, 3)
        atom_3_position = torch.gather(node_position14, -2,
                                       atom_3[:, None, None].expand(-1, -1, 3))  # (num_residue, 1, 3)
        axis = atom_3_position - atom_2_position
        axis_normalize = axis / (axis.norm(dim=-1, keepdim=True) + 1e-10)
        rotate_angle = rotate_angles[:, i, None, None]

        # Rotate all subsequent atoms by the rotation angle
        rotate_atoms_position = node_position14 - atom_2_position  # (num_residue, 14, 3)
        parallel_component = (rotate_atoms_position * axis_normalize).sum(dim=-1, keepdim=True) \
                             * axis_normalize
        perpendicular_component = rotate_atoms_position - parallel_component
        perpendicular_component_norm = perpendicular_component.norm(dim=-1, keepdim=True) + 1e-10
        perpendicular_component_normalize = perpendicular_component / perpendicular_component_norm
        normal_vector = torch.cross(axis_normalize.expand(-1, 14, -1), perpendicular_component_normalize, dim=-1)
        transformed_atoms_position = perpendicular_component * rotate_angle.cos() + \
                                     normal_vector * perpendicular_component_norm * rotate_angle.sin() + \
                                     parallel_component + atom_2_position  # (num_residue, 14, 3)
        assert not transformed_atoms_position.isnan().any()
        chi_mask = chi_atom14_mask[:, i, :].all(dim=-1, keepdim=True)  # (num_residue, 1)
        atom_mask = torch.arange(14, device=chi_mask.device)[None, :] >= atom_4[:, None]  # (num_residue, 14)
        mask = (atom_mask & chi_mask).unsqueeze(-1).expand_as(node_position14)
        node_position14[mask] = transformed_atoms_position[mask]

    protein.node_position[mask14] = node_position14[protein.atom2residue[mask14], protein.atom14index[mask14]]
    return chi_atom14_mask.all(dim=-1)

@torch.no_grad()
def set_chis(protein, chis):
    assert chis.shape[0] == protein.num_nodes
    assert chis.shape[1] == 4
    cur_chis = get_chis(protein)
    chi_to_rotate = chis - cur_chis
    chi_to_rotate[torch.isnan(chi_to_rotate)] = 0
    rotate_side_chain(protein, chi_to_rotate)
    return protein

def _get_symm_atoms(pos_per_residue, residue_type):
    sym_pos_per_residue = pos_per_residue.clone()   # [num_residues, 37, 3]
    for i in range(2):
        atom_to = res_sym_atom_posn.to(residue_type.device)[residue_type][:, i, 0]      # [num_residues]
        atom_from = res_sym_atom_posn.to(residue_type.device)[residue_type][:, i, 1]    # [num_residues]
        sym_pos_per_residue[torch.arange(len(pos_per_residue)), atom_to] = \
            pos_per_residue[torch.arange(len(pos_per_residue)), atom_from]
        sym_pos_per_residue[torch.arange(len(pos_per_residue)), atom_from] = \
            pos_per_residue[torch.arange(len(pos_per_residue)), atom_to]
    return sym_pos_per_residue

def _rmsd_per_residue(pred_pos_per_residue, true_pos_per_residue, mask):
    pred_pos_per_residue = pred_pos_per_residue[mask]
    true_pos_per_residue = true_pos_per_residue[mask]
    sd = torch.square(pred_pos_per_residue - true_pos_per_residue).sum(dim=-1)
    msd = sd.sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
    rmsd = msd.sqrt()
    return rmsd

