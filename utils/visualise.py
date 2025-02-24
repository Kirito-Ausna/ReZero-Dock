from rdkit.Chem.rdmolfiles import MolToPDBBlock, MolToPDBFile
import rdkit.Chem 
from rdkit import Geometry
from collections import defaultdict
import copy
import numpy as np
import torch
from datasets.process_mols import parse_pdb_from_path, extract_receptor_structure
from scipy import spatial
from utils.rotamer import atom_name_vocab
from Bio.PDB import PDBIO
import pdb
from rdkit.Chem import MolFromSmiles
from datasets.process_mols import read_molecule

    
class PDBFile:
    def __init__(self, mol):
        self.parts = defaultdict(dict)
        self.mol = copy.deepcopy(mol)
        [self.mol.RemoveConformer(j) for j in range(mol.GetNumConformers()) if j]        
    def add(self, coords, order, part=0, repeat=1):
        if type(coords) in [rdkit.Chem.Mol, rdkit.Chem.RWMol]:
            block = MolToPDBBlock(coords).split('\n')[:-2]
            self.parts[part][order] = {'block': block, 'repeat': repeat}
            return
        elif type(coords) is np.ndarray:
            coords = coords.astype(np.float64)
        elif type(coords) is torch.Tensor:
            coords = coords.double().numpy()
        for i in range(coords.shape[0]):
            self.mol.GetConformer(0).SetAtomPosition(i, Geometry.Point3D(coords[i, 0], coords[i, 1], coords[i, 2]))
        block = MolToPDBBlock(self.mol).split('\n')[:-2]
        self.parts[part][order] = {'block': block, 'repeat': repeat}
        
    def write(self, path=None, limit_parts=None):
        is_first = True
        str_ = ''
        for part in sorted(self.parts.keys()):
            if limit_parts and part >= limit_parts:
                break
            part = self.parts[part]
            keys_positive = sorted(filter(lambda x: x >=0, part.keys()))
            keys_negative = sorted(filter(lambda x: x < 0, part.keys()))
            keys = list(keys_positive) + list(keys_negative)
            for key in keys:
                block = part[key]['block']
                times = part[key]['repeat']
                for _ in range(times):
                    if not is_first:
                        block = [line for line in block if 'CONECT' not in line]
                    is_first = False
                    str_ += 'MODEL\n'
                    str_ += '\n'.join(block)
                    str_ += '\nENDMDL\n'
        if not path:
            return str_
        with open(path, 'w') as f:
            f.write(str_)

class ModifiedPDB:
    # implementation for save pdb file with modified pocket sidechain coordinates
    def __init__(self, pdb_path, ligand_description, pocket_pos) -> None:
        # self.pdb_path = pdb_path
        # self.mol = mol
        self.pocket_pos = pocket_pos
        prot = parse_pdb_from_path(pdb_path)
        self.prot = copy.deepcopy(prot)
        mol = MolFromSmiles(ligand_description)#NOTE: In crossdock setting, we need ligands binding to the holo structure
        if mol is None:
            mol = read_molecule(ligand_description, remove_hs=False, sanitize=True)
        self.mol = mol
        self.rec, self.coords, self.c_alpha_coords, self.n_coords, self.c_coords, _ = extract_receptor_structure(prot, self.mol)

    def to_pdb(self, out_path, pocket_only=False):
        sc_atom_idx = 0
        updated_atoms = {}
        skipped_atoms = []
        for atom in self.rec.get_atoms():
            if atom.name not in atom_name_vocab:
                skipped_atoms.append(atom)
                continue
            if pocket_only:
                try:
                    atom.set_coord(self.pocket_pos[sc_atom_idx])
                except:
                    pdb.set_trace()
            else:
                updated_atoms[atom] = self.pocket_pos[sc_atom_idx]
            sc_atom_idx += 1
        
        assert sc_atom_idx == len(self.pocket_pos), 'Not all sidechain atoms are updated, index may be mismatched.'
        # write to pdb file
        wirter = PDBIO()
        if pocket_only:
            wirter.set_structure(self.rec)
        else:
            modified_atom_num = 0
            for atom in self.prot.get_atoms():
                if atom in updated_atoms:
                    # if (atom.coord - updated_atoms[atom] + 1.257 != 0.0).any():
                    #     print('atom {} is modified from {} to {}'.format(atom.name, atom.coord, updated_atoms[atom]))
                    atom.set_coord(updated_atoms[atom])
                    modified_atom_num += 1
            assert modified_atom_num == len(updated_atoms.keys()), 'Not all sidechain atoms are modified, index may be mismatched.'
            wirter.set_structure(self.prot)

        wirter.save(out_path) 
            
    def old_to_pdb(self, out_path):
        # mimic process_mols.extract_receptor_structure, 
        # but change the coordinates of atoms while parsing in the same order of the preprocess. Just need two loops.
        conf = self.mol.GetConformer()
        lig_coords = conf.GetPositions()
        sc_atom_idx = 0
        for i, chain in enumerate(self.rec):
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
        # if sc_atom_idx != len(self.pocket_pos):
        #     print('Not all sidechain atoms are modified, index may be mismatched.')
        #     pdb.set_trace()
        wirter = PDBIO()
        wirter.set_structure(self.prot)
        wirter.save(out_path) 