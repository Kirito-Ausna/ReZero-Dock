import os
import torch
from Bio.PDB import PDBParser
from esm import FastaBatchedDataset, pretrained
from rdkit.Chem import AddHs, MolFromSmiles
from torch_geometric.data import Dataset, HeteroData
import esm
import glob
from datasets.process_mols import parse_pdb_from_path, generate_conformer, read_molecule, get_lig_graph_with_matching, \
    extract_receptor_structure, get_rec_graph, get_name_from_database, process_molecule_from_database
from datasets.process_mols import safe_index
from utils.rotamer import residue_list, atom_name_vocab, get_chi_mask, bb_atom_name, residue_list_expand
from utils import rotamer
import pdb
import copy

sidechian_features = ['chi_mask', 'chi_1pi_periodic_mask', 'chi_2pi_periodic_mask', 'atom37_mask', 'sidechain37_mask', 'atom14index', 'atom_name', 'atom2residue', 'residue_type', 'node_position', 'num_residue', 'num_nodes']

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
    def __init__(self, cache_dir, complex_names, protein_files, ligand_descriptions, end_ligand_id,
                 protein_target_path, ligand_in_pocket_path, ligand_database_path, start_ligand_id,
                 protein_sequences, mode, receptor_radius, remove_hs, out_dir, num_workers,
                 c_alpha_max_neighbors=None, all_atoms=True, atom_radius=5, atom_max_neighbors=None):
        super(InferenceDatasets, self).__init__()
        self.cache_dir = cache_dir
        self.mode = mode
        # for modes except virtual screening, we need to provide the protein and ligand files
        self.complex_names = complex_names
        self.protein_files = protein_files
        self.ligand_descriptions = ligand_descriptions
        self.protein_sequences = protein_sequences
        # for virtual screening
        if self.mode == 'virtual_screen':
            self.protein_target_path = protein_target_path
            self.ligand_in_pocket_path = ligand_in_pocket_path
            self.ligand_database_path = ligand_database_path
            self.start_ligand_id = start_ligand_id
            self.end_liand_id = end_ligand_id
            self.ligand_names = get_name_from_database(ligand_database_path, start_ligand_id, end_ligand_id)

        self.out_dir = out_dir

        self.num_workers = num_workers
        self.receptor_radius = receptor_radius
        self.remove_hs = remove_hs
        self.c_alpha_max_neighbors = c_alpha_max_neighbors
        self.all_atoms = all_atoms
        self.atom_radius = atom_radius
        self.atom_max_neighbors = atom_max_neighbors

        self.if_cache = True if self.cache_dir else False

        if self.if_cache and self.mode != 'virtual_screen': # we save and read esm embeddings from cache
            self.target_name = os.path.basename(self.out_dir)
            target_cache_dir = os.path.join(cache_dir, self.target_name)
            if not os.path.exists(target_cache_dir):
                os.makedirs(target_cache_dir)
            self.lig_graph_cache = os.path.join(target_cache_dir, 'rdkit_mols_graph.pt')
            self.rec_graph_cache = os.path.join(target_cache_dir, 'rec_graph.pt')
            self.mol_cache = os.path.join(target_cache_dir, 'mol_dict.pt')
        
        if self.if_cache and self.mode == 'virtual_screen':
            self.target_name = os.path.basename(self.protein_target_path)[:-4]
            self.database_name = os.path.basename(self.ligand_database_path)[:-4]
            target_cache_dir = os.path.join(cache_dir, self.target_name)
            ligand_database_cache_dir = os.path.join(cache_dir, self.database_name)
            if not os.path.exists(target_cache_dir):
                os.makedirs(target_cache_dir)
            if not os.path.exists(ligand_database_cache_dir):
                os.makedirs(ligand_database_cache_dir)
            # self.lig_graph_cache = os.path.join(ligand_database_cache_dir, f'rdkit_mols_graph_{self.end_liand_id}.pt')
            self.lig_graph_cache = os.path.join(ligand_database_cache_dir, f'rdkit_mols_graph.pt')
            self.rec_graph_cache = os.path.join(target_cache_dir, 'rec_graph.pt')
            # self.mol_cache = os.path.join(ligand_database_cache_dir, f'mol_dict_{self.end_liand_id}.pt')
            self.mol_cache = os.path.join(ligand_database_cache_dir, f'mol_dict.pt')

        # disentangle the protein and ligand graphs
        self.lig_graph_dict = None
        self.rec_graph_dict = None
        self.mol_dict = None # for saving the rdkit conformation and meta data of the molecules 

        # read from cache if available
        # pdb.set_trace()
        loaded = 0
        if self.if_cache and os.path.exists(self.rec_graph_cache):
            self.rec_graph_dict = torch.load(self.rec_graph_cache)
            print('loaded receptor graphs from cache')
            loaded += 1

        if self.if_cache and os.path.exists(self.lig_graph_cache):
            self.lig_graph_dict = torch.load(self.lig_graph_cache)
            print('loaded ligand graphs from cache')
            self.mol_dict = torch.load(self.mol_cache)
            print('loaded mol dict from cache')
            loaded += 1
        # generate LM embeddings and cache them, organize them by protein name
        if loaded < 2:
            self.preprocess() # generate esm embeddings, ligand graphs, receptor graphs mol dict according to the mode

    def len(self):
        if self.mode != 'virtual_screen':
            return len(self.complex_names)
        return len(self.ligand_names)
    
    def copy_graph(self, src_graph, dst_graph):
        # copy all the node and edge features
        # copy the edge index
        for key in src_graph.edge_types:
            dst_graph[key] = src_graph[key]
        # copy the node features
        for key in src_graph.node_types:
            dst_graph[key] = src_graph[key]
        # pdb.set_trace()
        # dst_graph.update(src_graph)
        # src_graph.update(dst_graph)
        return dst_graph
        # pdb.set_trace()

    def construct_complex_graph(self, lig_graph, rec_graph, complex_graph):
        complex_graph.rmsd_matching = 0
        complex_graph['ligand'].edge_mask = lig_graph['ligand'].edge_mask
        complex_graph['ligand'].mask_rotate = lig_graph['ligand'].mask_rotate
        
        complex_graph['ligand'].x = lig_graph['ligand'].x
        complex_graph['ligand'].pos = lig_graph['ligand'].pos
        complex_graph['ligand','lig_bond', 'ligand'].edge_index = lig_graph['ligand','lig_bond', 'ligand'].edge_index
        complex_graph['ligand','lig_bond', 'ligand'].edge_attr = lig_graph['ligand','lig_bond', 'ligand'].edge_attr

        complex_graph['receptor'].x = rec_graph['receptor'].x
        complex_graph['receptor'].pos = rec_graph['receptor'].pos
        complex_graph['receptor'].mu_r_norm = rec_graph['receptor'].mu_r_norm
        complex_graph['receptor'].side_chain_vecs = rec_graph['receptor'].side_chain_vecs
        complex_graph['receptor','rec_contact', 'receptor'].edge_index = rec_graph['receptor','rec_contact', 'receptor'].edge_index
        complex_graph['atom'].x = rec_graph['atom'].x
        complex_graph['atom'].pos = rec_graph['atom'].pos
        complex_graph['atom','atom_contact', 'atom'].edge_index = rec_graph['atom','atom_contact', 'atom'].edge_index
        complex_graph['atom','atom_rec_contact', 'receptor'].edge_index = rec_graph['atom','atom_rec_contact', 'receptor'].edge_index

        for feat in sidechian_features:
            complex_graph['sidechain'][feat] = rec_graph['sidechain'][feat]

        # return complex_graph

    def get(self, idx):
        if self.mode == 'crossdock':
            complex_name = self.complex_names[idx]
            protein_name = complex_name.split('_')[0]
            ligand_name = complex_name.split('_')[2]
        elif self.mode == 'virtual_screen':
            ligand_name = self.ligand_names[idx] # get ligand id for the database
            protein_name = self.target_name # target is fixed in virtual screening
            complex_name = protein_name + '_' + ligand_name # create a complex name for the virtual screening and saved folder name     
        elif self.mode == 'apodock':
            complex_name = self.complex_names[idx]
            # convert complex_name to string type
            complex_name = str(complex_name)
            # get the protein name form self.protein_files #TODO: unifed protein name and ligand name parser. Currenly, we need to strictly follow the naming rule of the dataset.
            protein_name = os.path.basename(self.protein_files[idx]).split('_')[0]
            ligand_name = complex_name+'.sdf'
            # ligand_name = ligand_name.split('_')[0]
        # bulid the heterograph
        complex_graph = HeteroData()
        complex_graph.name = complex_name
        if self.mode != 'virtual_screen': # for saving pokcet results
            complex_graph.prot_path = self.protein_files[idx]
            complex_graph.lig_path = self.ligand_descriptions[idx]

        # check if exists and add ligand graph
        if ligand_name in self.lig_graph_dict and protein_name in self.rec_graph_dict:
            lig_graph = copy.deepcopy(self.lig_graph_dict[ligand_name])
            rec_graph = copy.deepcopy(self.rec_graph_dict[protein_name])
            self.construct_complex_graph(lig_graph, rec_graph, complex_graph)
            # complex_graph = self.copy_graph(lig_graph, complex_graph)
            # complex_graph.update(lig_graph)
        else:
            print('ligand or receptor graph not found for', complex_name)
            complex_graph['success'] = False
            # pdb.set_trace()
            return complex_graph

        # check if exists and add receptor graph
        # if protein_name in self.rec_graph_dict:
        #     rec_graph = self.rec_graph_dict[protein_name]
        #     # complex_graph['receptor'] = rec_graph
        #     # complex_graph = self.copy_graph(rec_graph, complex_graph)
        #     # complex_graph.update(rec_graph)
        # else:
        #     print('receptor graph not found for', complex_name)
        #     complex_graph['success'] = False
        #     return complex_graph
        # pdb.set_trace()
        protein_center = torch.mean(complex_graph['receptor'].pos, dim=0, keepdim=True)
        complex_graph['receptor'].pos -= protein_center
        if self.all_atoms:
            complex_graph['atom'].pos -= protein_center

        ligand_center = torch.mean(complex_graph['ligand'].pos, dim=0, keepdim=True)
        complex_graph['ligand'].pos -= ligand_center

        complex_graph.original_center = protein_center
        complex_graph.mol = self.mol_dict[ligand_name] #NOTE: We need mol for ligand meta information for saving the results
        # complex_graph.mol = read_molecule(self.ligand_descriptions[idx], remove_hs=False, sanitize=True)
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
        assert protein.num_residue == protein.residue_type.shape[0] # prevent the bug of different residue number when meeting trash data
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
        if self.mode != 'virtual_screen':
            distinct_protein_paths = list(set(self.protein_files))
            distinct_ligand_paths = list(set(self.ligand_descriptions))
            protein_names = [os.path.basename(protein_path).split('_')[0] for protein_path in distinct_protein_paths]
            ligand_names = [os.path.basename(ligand_path).split('_')[0] for ligand_path in distinct_ligand_paths]
        else:
            protein_names = [self.target_name]
            distinct_protein_paths = [self.protein_target_path]
            distinct_ligand_paths = [self.ligand_in_pocket_path]
            ligand_names = self.ligand_names
            
        if not os.path.exists(self.rec_graph_cache):
            # generate esm embeddings
            lm_embeddings_chains_all = {}
            if self.cache_dir:
                esm_embedding_path = os.path.join(self.cache_dir, 'esm_embeddings') # check if the embeddings are already computed
            if self.cache_dir and os.path.exists(esm_embedding_path):
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

        if not os.path.exists(self.lig_graph_cache):
            # generate mol dict
            mol_dict = {}
            if self.cache_dir:
                rdkit_mol_path = os.path.join(self.cache_dir, 'rdkit_mols')
            if self.cache_dir and os.path.exists(rdkit_mol_path): #NOTE: We encourage to generate rdkit conformers in advance
                print('loading rdkit mols from', rdkit_mol_path)
                for ligand_name in ligand_names:
                    try:
                        path = os.path.join(rdkit_mol_path, ligand_name + '.sdf')
                        if not os.path.exists(path):
                            print('rdkit mol not found for', ligand_name)
                            mol_dict[ligand_name] = None
                            continue
                        mol_dict[ligand_name] = read_molecule(path, remove_hs=False, sanitize=True)
                    except Exception as e:
                        print('error loading rdkit mol for', ligand_name)
                        print(e)
                        mol_dict[ligand_name] = None # will be skipped in ligand graph generation
            else:
                print("Generating rdkit mols")
                if self.mode != 'virtual_screen':
                    for i, ligand_description in enumerate(distinct_ligand_paths):
                        mol = read_molecule(ligand_description, remove_hs=False, sanitize=True)
                        if mol is None:
                            print('None mol for', ligand_description)
                            mol_dict[ligand_names[i]] = None
                            continue
                        try:
                            mol.RemoveAllConformers()
                            mol = AddHs(mol)
                            generate_conformer(mol)
                            mol_dict[ligand_names[i]] = mol
                        except:
                            print('rdkit conformer generation failed for', ligand_names[i])
                            mol_dict[ligand_names[i]] = None
                else: #TODO: support parralel rdkit conformer generation in preprocess
                    database_mols = process_molecule_from_database(self.ligand_database_path, self.start_ligand_id)
                    for ligand_name, rdkit_mol in database_mols:
                        mol_dict[ligand_name] = rdkit_mol

            if self.if_cache:
                torch.save(mol_dict, self.mol_cache)
                print('saved mol dict to', self.mol_cache)
            self.mol_dict = mol_dict
            # generate ligand graphs
            lig_graph_dict = {}
            print('Generating ligand graphs')
            for lig_name in ligand_names:
                lig_graph = HeteroData()
                if mol_dict[lig_name] is None:
                    print('rdkit conf unfound, skipping ligand graph generation for', lig_name)
                    continue
                try:
                    get_lig_graph_with_matching(mol_dict[lig_name], lig_graph, popsize=None, maxiter=None, matching=False, keep_original=False,
                                            num_conformers=1, remove_hs=self.remove_hs)
                    lig_graph_dict[lig_name] = lig_graph
                except Exception as e:
                    print('ligand graph generation failed for', lig_name)
                    print(e)
                    continue
            if self.if_cache:
                torch.save(lig_graph_dict, self.lig_graph_cache)
                print('saved ligand graphs to', self.lig_graph_cache)
            self.lig_graph_dict = lig_graph_dict
        
        if not os.path.exists(self.rec_graph_cache):
        # generate receptor graphs
            rec_graph_dict = {}
            print('Generating receptor graphs')
            for i, protein_name in enumerate(protein_names):
                # pdb.set_trace()
                protein_path = distinct_protein_paths[i]
                rec_graph = HeteroData()
                rec_model = parse_pdb_from_path(distinct_protein_paths[i])
                # find the corresponding ligand in holo structure according to the protein name
                # for ligand_description in distinct_ligand_paths:
                #     if protein_name in ligand_description:
                #         break
                holo_ligand_name = os.path.basename(protein_path).split('.')[0].replace('PRO', 'LIG')
                ligand_description = os.path.join(os.path.dirname(protein_path), holo_ligand_name+'.sdf')
                true_mol = read_molecule(ligand_description, remove_hs=False, sanitize=True)
                # if self.mode != 'virtual_screen': # protein and ligand share the same key
                try: #NOTE: the pocket information is based on known binding ligand
                    # pdb.set_trace()
                    rec, rec_coords, c_alpha_coords, n_coords, c_coords, lm_embeddings = extract_receptor_structure(rec_model, true_mol, 
                                                                                                                    lm_embedding_chains=lm_embeddings_chains_all[protein_name])
                    if lm_embeddings is not None and len(c_alpha_coords) != len(lm_embeddings):
                        print('protein and esm embeddings have different lengths')
                        continue
                    get_rec_graph(rec, rec_coords, c_alpha_coords, n_coords, c_coords, rec_graph, rec_radius=self.receptor_radius,
                            c_alpha_max_neighbors=self.c_alpha_max_neighbors, all_atoms=self.all_atoms,
                            atom_radius=self.atom_radius, atom_max_neighbors=self.atom_max_neighbors, remove_hs=self.remove_hs, lm_embeddings=lm_embeddings)
                    self.chi_torsion_features(rec_graph, rec) # sidechain graph as part of the receptor graph
                    rec_graph_dict[protein_name] = rec_graph
                # pdb.set_trace()
                except Exception as e:
                    print('receptor graph generation failed for', protein_name)
                    print(e)
                    continue
                
            if self.if_cache:
                torch.save(rec_graph_dict, self.rec_graph_cache)
                print('saved receptor graphs to', self.rec_graph_cache)
            self.rec_graph_dict = rec_graph_dict


                

        