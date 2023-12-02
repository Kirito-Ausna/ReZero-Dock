import os
import torch
from Bio.PDB import PDBParser
from esm import FastaBatchedDataset, pretrained
from rdkit.Chem import AddHs, MolFromSmiles
from torch_geometric.data import Dataset, HeteroData
import esm
import glob
from datasets.process_mols import parse_pdb_from_path, generate_conformer, read_molecule, get_lig_graph_with_matching, \
    extract_receptor_structure, get_rec_graph
from datasets.process_mols import safe_index
from utils.rotamer import residue_list, atom_name_vocab, get_chi_mask, bb_atom_name, residue_list_expand
from utils import rotamer
import pdb

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


def set_nones(l):
    return [s if str(s) != 'nan' else None for s in l]


def get_sequences(protein_files, protein_sequences):
    new_sequences = []
    for i in range(len(protein_files)):
        if protein_files[i] is not None:
            new_sequences.append(get_sequences_from_pdbfile(protein_files[i]))
        else:
            new_sequences.append(protein_sequences[i]) #TODO: don't support crossdock mode
    return new_sequences


def compute_ESM_embeddings(model, alphabet, labels, sequences):
    # settings used
    toks_per_batch = 4096
    repr_layers = [33]
    include = "per_tok"
    truncation_seq_length = 1022

    dataset = FastaBatchedDataset(labels, sequences)
    batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=alphabet.get_batch_converter(truncation_seq_length), batch_sampler=batches
    )

    assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in repr_layers)
    repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in repr_layers]
    embeddings = {}

    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            print(f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)")
            if torch.cuda.is_available():
                toks = toks.to(device="cuda", non_blocking=True)

            out = model(toks, repr_layers=repr_layers, return_contacts=False)
            representations = {layer: t.to(device="cpu") for layer, t in out["representations"].items()}

            for i, label in enumerate(labels):
                truncate_len = min(truncation_seq_length, len(strs[i]))
                embeddings[label] = representations[33][i, 1: truncate_len + 1].clone()
    return embeddings

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
                print("saved", filename)
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

class InferenceDatasets(Dataset):
    def __init__(self, cache_dir, complex_names, protein_files, ligand_descriptions,
                 protein_sequences, mode, receptor_radius, remove_hs, out_dir, num_workers,
                 c_alpha_max_neighbors=None, all_atoms=True, atom_radius=5, atom_max_neighbors=None):
        self.cache_dir = cache_dir
        self.complex_names = complex_names
        self.protein_files = protein_files
        self.ligand_descriptions = ligand_descriptions
        self.protein_sequences = protein_sequences
        self.mode = mode
        self.out_dir = out_dir

        self.num_workers = num_workers
        self.receptor_radius = receptor_radius
        self.remove_hs = remove_hs
        self.c_alpha_max_neighbors = c_alpha_max_neighbors
        self.all_atoms = all_atoms
        self.atom_radius = atom_radius
        self.atom_max_neighbors = atom_max_neighbors

        if self.cache_dir is None or self.mode == 'vitual_screening':
            self.if_cache = False
        else: self.if_cache = True
        if self.if_cache: # we save and read esm embeddings from cache
            self.lig_graph_cache = os.path.join(cache_dir, 'rdkit_mols_graph.pt')
            self.rec_graph_cache = os.path.join(cache_dir, 'rec_graph.pt')
            self.mol_cache = os.path.join(cache_dir, 'mol_dict.pt')

        # disentangle the protein and ligand graphs
        self.lig_graph_dict = None
        self.rec_graph_dict = None
        self.mol_dict = None # for saving the meta data of the molecules 

        # read from cache if available
        if self.if_cache and os.path.exists(self.lig_graph_cache):
            self.lig_graph_dict = torch.load(self.lig_graph_cache)
            print('loaded ligand graphs from cache')
            self.rec_graph_dict = torch.load(self.rec_graph_cache)
            print('loaded receptor graphs from cache')
            self.mol_dict = torch.load(self.mol_cache)
            print('loaded mol dict from cache')
        # generate LM embeddings and cache them, organize them by protein name
        else:
            self.preprocess() # generate esm embeddings, ligand graphs, receptor graphs mol dict according to the mode
    def __len__(self):
        return len(self.complex_names)
    
    def copy_graph(self, src_graph, dst_graph):
        # copy all the node and edge features
        # copy the edge index
        for key in src_graph.edge_index_dict.keys():
            dst_graph[key] = src_graph[key]
        # copy the node features
        for key in src_graph.node_attr_dict.keys():
            dst_graph[key] = src_graph[key]
        # copy the node position
        for key in src_graph.node_pos_dict.keys():
            dst_graph[key] = src_graph[key]

    def __getitem__(self, idx):
        complex_name = self.complex_names[idx]
        protein_name = complex_name.split('_')[0]
        ligand_name = complex_name.split('_')[3]

        # bulid the heterograph
        complex_graph = HeteroData()
        complex_graph.name = complex_name
        complex_graph.prot_path = self.protein_files[idx]
        complex_graph.lig_path = self.ligand_descriptions[idx]

        # check if exists and add ligand graph
        if ligand_name in self.lig_graph_dict:
            lig_graph = self.lig_graph_dict[ligand_name]
            self.copy_graph(lig_graph, complex_graph)
        else:
            print('ligand graph not found for', complex_name)
            complex_graph['suceess'] = False
            return complex_graph

        # check if exists and add receptor graph
        if protein_name in self.rec_graph_dict:
            rec_graph = self.rec_graph_dict[protein_name]
            # complex_graph['receptor'] = rec_graph
            self.copy_graph(rec_graph, complex_graph)
        else:
            print('receptor graph not found for', complex_name)
            complex_graph['suceess'] = False
            return complex_graph
        protein_center = torch.mean(complex_graph['receptor'].pos, dim=0, keepdim=True)
        complex_graph['receptor'].pos -= protein_center
        if self.all_atoms:
            complex_graph['atom'].pos -= protein_center

        ligand_center = torch.mean(complex_graph['ligand'].pos, dim=0, keepdim=True)
        complex_graph['ligand'].pos -= ligand_center

        complex_graph.original_center = protein_center
        complex_graph.mol = self.mol_dict[ligand_name]
        complex_graph['success'] = True
        return complex_graph
    
    # functions for sidechain torsional diffusion
    def chi_torsion_features(self, complex_graphs, rec):
        # get residue type and atom name lists [Num_atom] * 2
        # atom_feat = comlex_graphs['atom'].x
        # atom_residue_type = atom_feat[:, 0]
        # NOTE: The index system is different from the one in process_mols.py
        atom_residue_type = [] # 20-indexed number as AlphaFold [Num_atom,]
        atom_name = [] # 37-indexed fixed number [Num_atom,]
        atom2residue = [] # [Num_atom,] convert the atom_residue_type to residue index array [Num_atoms,] like [0,0,...,0,1,1,..1,...]
        res_index = 0
        last_residue = None
        residue_type = []
        for i, atom in enumerate(rec.get_atoms()):  # NOTE: atom index must be the same as the one in process_mols.py
            if atom.name not in atom_name_vocab:
                continue
            res = atom.get_parent()
            res_type_check = safe_index(residue_list_expand, res.get_resname())
            if res_type_check > 20:
                res_type = 0
            else:
                res_type = safe_index(residue_list, res.get_resname()) # make sure it's normal amino acid
            res_id = res.get_id() # A residue id is a tuple of (hetero-flag, sequence identifier, insertion code)
            # pdb.set_trace()
            atom_name.append(atom_name_vocab[atom.name])
            atom_residue_type.append(res_type)
            if last_residue is None:
                last_residue = res_id
                residue_type.append(res_type)
            atom2residue.append(res_index)
            if res_id != last_residue:
                res_index += 1
                last_residue = res_id
                residue_type.append(res_type)
        protein = complex_graphs['sidechain']
        # save the rec_coords for usage in add_noise
        # protein.rec_coords = rec_coords # the rec_coords here include H atoms
        atom_residue_type = torch.tensor(atom_residue_type)
        atom_name = torch.tensor(atom_name)
        # pdb.set_trace()
        protein.atom14index = rotamer.restype_atom14_index_map[atom_residue_type, atom_name] # [num_atom,]
        protein.atom_name = atom_name
        # complex_graphs['receptor'].residue_type = atom_residue_type
        residue_indexes = complex_graphs['receptor'].x[:,0]
        protein.num_residue = residue_indexes.shape[0]
        protein.num_nodes = complex_graphs['receptor'].num_nodes
        # complex_graphs['sidechain'].num_nodes = complex_graphs['receptor'].num_nodes
        protein.atom2residue = torch.tensor(atom2residue)
        protein.residue_type = torch.tensor(residue_type)
        atom_position = complex_graphs['atom'].pos #NOTE: [num_atom, 3] and the atom index must be the same as the one in process_mols.py
        protein.node_position = atom_position
        # Init residue masks
        chi_mask = get_chi_mask(protein, device=atom_position.device) # [num_residue, 4]
        # pdb.set_trace()
        chi_1pi_periodic_mask = torch.tensor(rotamer.chi_pi_periodic)[protein.residue_type]
        chi_2pi_periodic_mask = ~chi_1pi_periodic_mask
        protein.chi_mask = chi_mask
        protein.chi_1pi_periodic_mask = torch.logical_and(chi_mask, chi_1pi_periodic_mask)  # [num_residue, 4]
        protein.chi_2pi_periodic_mask = torch.logical_and(chi_mask, chi_2pi_periodic_mask)  # [num_residue, 4]
        # Init atom37 features
        protein.atom37_mask = torch.zeros(protein.num_residue, len(atom_name_vocab), device=chi_mask.device,
                                            dtype=torch.bool)  # [num_residue, 37]
        protein.atom37_mask[protein.atom2residue, protein.atom_name] = True
        protein.sidechain37_mask = protein.atom37_mask.clone()  # [num_residue, 37]
        protein.sidechain37_mask[:, bb_atom_name] = False
        
    def preprocess(self): #NOTE: Only support protein and liand files as input, protein seq and smiles are supported by ReDock_evalaute.py
        distinct_protein_paths = list(set(self.protein_files))
        distinct_ligand_paths = list(set(self.ligand_descriptions))
        protein_names = [os.path.basename(protein_path).split('_')[0] for protein_path in distinct_protein_paths]
        ligand_names = [os.path.basename(ligand_path).split('_')[0] for ligand_path in distinct_ligand_paths] 
        # generate esm embeddings
        lm_embeddings_chains_all = {}
        esm_embedding_path = os.path.join(self.cache_dir, 'esm_embeddings') # check if the embeddings are already computed
        if os.path.exists(esm_embedding_path):
            print('loading esm embeddings from', esm_embedding_path)
            for protein_name in protein_names:
                embeddings_paths = sorted(glob.glob(os.path.join(esm_embedding_path, protein_name) + '*'))
                lm_embeddings_chains = []
                for embeddings_path in embeddings_paths:
                    lm_embeddings_chains.append(torch.load(embeddings_path)['representations'][33])
                lm_embeddings_chains_all[protein_name] = torch.cat(lm_embeddings_chains)
        else:
            print("Generating ESM language model embeddings")
            model_location = "esm2_t33_650M_UR50D"
            model, alphabet = pretrained.load_model_and_alphabet(model_location)
            model.eval()
            if torch.cuda.is_available():
                model = model.cuda()

            protein_sequences = get_sequences(distinct_protein_paths, self.protein_sequences)
            labels, sequences = [], []
            for i in range(len(protein_sequences)):
                s = protein_sequences[i].split(':')
                sequences.extend(s)
                labels.extend([protein_names[i] + '_chain_' + str(j) for j in range(len(s))])

            lm_embeddings = compute_ESM_embeddings(model, alphabet, labels, sequences)

            for i in range(len(protein_sequences)):
                s = protein_sequences[i].split(':')
                lm_embeddings_chains_all[protein_names[i]] = [lm_embeddings[f'{protein_names[i]}_chain_{j}'] for j in range(len(s))]

        # generate mol dict
        mol_dict = {}
        rdkit_mol_path = os.path.join(self.cache_dir, 'rdkit_mols')
        if os.path.exists(rdkit_mol_path):
            print('loading rdkit mols from', rdkit_mol_path)
            for ligand_name in ligand_names:
                mol_dict[ligand_name] = read_molecule(os.path.join(rdkit_mol_path, ligand_name + '.sdf'), 
                                                      remove_hs=False, sanitize=True)
        else:
            print("Generating rdkit mols")
            for i, ligand_description in enumerate(distinct_ligand_paths):
                mol = read_molecule(ligand_description, remove_hs=False, sanitize=True)
                if mol is None:
                    print('None mol for', ligand_description)
                    continue
                mol.RemoveAllConformers()
                mol = AddHs(mol)
                generate_conformer(mol)
                mol_dict[ligand_names[i]] = mol
            if self.if_cache:
                torch.save(mol_dict, self.mol_cache)

        # generate ligand graphs
        lig_graph_dict = {}
        for lig_name in ligand_names:
            lig_graph = HeteroData()
            try:
                get_lig_graph_with_matching(mol_dict[ligand_name], lig_graph, popsize=None, maxiter=None, matching=False, keep_original=False,
                                        num_conformers=1, remove_hs=self.remove_hs)
                lig_graph_dict[lig_name] = lig_graph
            except Exception as e:
                print('ligand graph generation failed for', lig_name)
                print(e)
                continue
        if self.if_cache:
            torch.save(lig_graph_dict, self.lig_graph_cache)
        # generate receptor graphs
        rec_graph_dict = {}
        for i, protein_name in enumerate(protein_names):
            rec_graph = HeteroData()
            rec_model = parse_pdb_from_path(distinct_protein_paths[i])
            # if self.mode != 'virtual_screen': # protein and ligand share the same key
            try: #NOTE: the pocket information is based on known binding ligand
                rec, rec_coords, c_alpha_coords, n_coords, c_coords, lm_embeddings = extract_receptor_structure(rec_model, mol_dict[protein_name], 
                                                                                                                lm_embedding_chains=lm_embeddings_chains_all[protein_name])
                if lm_embeddings is not None and len(c_alpha_coords) != len(lm_embeddings):
                    print('protein and esm embeddings have different lengths')
                    continue
                get_rec_graph(rec, rec_coords, c_alpha_coords, n_coords, c_coords, rec_graph, rec_radius=self.receptor_radius,
                        c_alpha_max_neighbors=self.c_alpha_max_neighbors, all_atoms=self.all_atoms,
                        atom_radius=self.atom_radius, atom_max_neighbors=self.atom_max_neighbors, remove_hs=self.remove_hs, lm_embeddings=lm_embeddings)
                rec_graph_dict[protein_name] = rec_graph
                self.chi_torsion_features(rec_graph, rec) # sidechain graph as part of the receptor graph
            except Exception as e:
                print('receptor graph generation failed for', protein_name)
                print(e)
                continue
        if self.if_cache:
            torch.save(rec_graph_dict, self.rec_graph_cache)
            


                

        