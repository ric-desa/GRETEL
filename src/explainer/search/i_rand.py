import random
import itertools
import numpy as np
import copy
import torch
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
import os
import networkx as nx, matplotlib.pyplot as plt


from src.core.explainer_base import Explainer
from src.dataset.instances.graph import GraphInstance


class IRandExplainer(Explainer):
    """iRand stands for Iterative Random Explainer, the logic of the explainers is to 
    """
            
    def init(self):
        super().init()

        self.perturbation_percentage = self.local_config['parameters']['p']
        self.tries = self.local_config['parameters']['t']
        self.visualize = self.local_config['parameters'].get('visualize', False)
        self.vis_id = self.local_config['parameters'].get('vis_id', -1)
        self.chem_flag = self.local_config['parameters'].get('chem_flag', False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def explain(self, instance):
        # counter = 0
        # while True:
            if self.visualize and instance.id != self.vis_id and self.vis_id != -1:
                return instance
            
            # counter += 1
            # print(f"\rCounter: {counter}", end="", flush=True)
                    
            # self.data = torch.tensor(instance.data, dtype=torch.double, device=self.device)
            # self.batch = torch.zeros(instance.node_features.shape[0], dtype=torch.long, device=self.device)
            # edge_indices = torch.where(self.data != 0) # (int tensor)
            # edge_weights = self.data.detach().clone()[edge_indices[0], edge_indices[1]]
            # self.adj_full = torch.ones_like(self.data)
            # edge_weights_full = torch.zeros_like(self.data) 
            # edge_weights_full[edge_indices] = edge_weights # weights having also 0s for missing edges
            # edge_weights_full = edge_weights_full.flatten()
            # edge_index_full = self.adj_full.nonzero(as_tuple=False).T
            # l_input_inst = torch.argmax(self.oracle.model(torch.tensor(instance.node_features, dtype=torch.float64, device=self.device),edge_index_full,edge_weights_full,self.batch).clone().detach(), dim=-1).squeeze(-1) # Get GCN prediction
            # self.oracle._call_counter += 1

            
            l_input_inst = self.oracle.predict(instance)
            nodes = instance.data.shape[0]

            # all edges (direct graph)
            all_edges = list(itertools.product(list(range(nodes)), repeat=2))
            # filter for only undirected edges
            new_edges = list()
            for edge in all_edges:
                if ((edge[1], edge[0]) not in new_edges) and edge[0] != edge[1]:
                    new_edges.append(list(edge))
            new_edges = np.array(new_edges)
            
            # Calculate the maximun percent of edges to modify
            k = int(len(new_edges) * self.perturbation_percentage)

            # increase the number of random 
            result = None
            for i in range(1, k):
                # how many attempts at a current modification level
                for j in range(0, self.tries):
                    cf_cand_matrix = np.copy(instance.data)
                    # sample according to perturbation_percentage
                    sample_index = np.random.choice(list(range(len(new_edges))), size=i)
                    sampled_edges = new_edges[sample_index]

                    # switch on/off the sampled edges
                    cf_cand_matrix[sampled_edges[:,0], sampled_edges[:,1]] = 1 - cf_cand_matrix[sampled_edges[:,0], sampled_edges[:,1]]
                    cf_cand_matrix[sampled_edges[:,1], sampled_edges[:,0]] = 1 - cf_cand_matrix[sampled_edges[:,1], sampled_edges[:,0]]
                
                    # build the counterfactaul candidates instance
                    result = GraphInstance(id=instance.id,
                                        label=0,
                                        data=cf_cand_matrix,
                                        node_features=instance.node_features)
                    
                    # if a counterfactual was found return that

                    # result_data = torch.tensor(result.data, dtype=torch.double, device=self.device)
                    # result_batch = torch.zeros(result.node_features.shape[0], dtype=torch.long, device=self.device)
                    # result_edge_indices = torch.where(result_data != 0) # (int tensor)
                    # result_edge_weights = result_data.detach().clone()[result_edge_indices[0], result_edge_indices[1]]
                    # result_adj_full = torch.ones_like(result_data)
                    # result_edge_weights_full = torch.zeros_like(result_data) 
                    # result_edge_weights_full[result_edge_indices] = result_edge_weights # weights having also 0s for missing edges
                    # result_edge_weights_full = result_edge_weights_full.flatten()
                    # result_edge_index_full = result_adj_full.nonzero(as_tuple=False).T
                    # l_cf_cand = torch.argmax(self.oracle.model(torch.tensor(instance.node_features, dtype=torch.float64, device=self.device),result_edge_index_full,result_edge_weights_full,result_batch).clone().detach(), dim=-1).squeeze(-1) # Get GCN prediction
                    # self.oracle._call_counter += 1

                    l_cf_cand = self.oracle.predict(result)
                    if l_input_inst != l_cf_cand:
                        result.label = l_cf_cand

                        if self.visualize:
                            pos = nx.spring_layout(nx.from_numpy_array(instance.data)) # Fix graph orientation
                            instance_graph = nx.from_numpy_array(instance.data)
                            CF_graph = nx.from_numpy_array(result.data)
                            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                            nx.draw(instance_graph, pos=pos, ax=axes[0], with_labels=True, cmap='cool', node_color=instance.node_features.mean(axis=1), edge_color='gray')
                            axes[0].set_title(f"Initial Graph | Predicted Class: {l_input_inst}")
                            nx.draw(CF_graph, pos=pos, ax=axes[1], with_labels=True, cmap='cool', node_color=result.node_features.mean(axis=1), edge_color='gray')
                            axes[1].set_title(f"Counterfactual Graph | Predicted Class: {result.label}")
                            fig.suptitle(f"True label: {instance.label}")
                            plt.show()

                            save = input("save graph? (y/n): ")
                        
                            if save.lower() == "y":

                                oracle_name = str(self.oracle.model.__class__).split('.')[-2]
                                G = nx.from_numpy_array(result.data)

                                for ix, feat in enumerate(result.node_features):
                                    # print(feat.mean())
                                    G.nodes[ix]["Feature"] = feat.mean()

                                nx.write_gexf(G, f"CFs_figs\\{instance.id}-iRand-{oracle_name}-cf.gexf")
                                input(f"Graph saved to CFs_figs\{instance.id}-iRand-{oracle_name}-cf.gexf")

                        return result
                    # print(cf_cand_matrix)

            if result and self.visualize:
                # If no counterfactual was found return the original instance by convention
                pos = nx.spring_layout(nx.from_numpy_array(instance.data)) # Fix graph orientation
                instance_graph = nx.from_numpy_array(instance.data)
                CF_graph = nx.from_numpy_array(cf_cand_matrix)
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                nx.draw(instance_graph, pos=pos, ax=axes[0], with_labels=True, cmap='cool', node_color=instance.node_features.mean(axis=1), edge_color='gray')
                axes[0].set_title(f"Initial Graph | Predicted Class: {l_input_inst}")
                nx.draw(CF_graph, pos=pos, ax=axes[1], with_labels=True, cmap='cool', node_color=result.node_features.mean(axis=1), edge_color='gray')
                axes[1].set_title(f"Counterfactual Graph | Predicted Class: {instance.label}")
                fig.suptitle(f"True label: {instance.label}")
                plt.show()

                save = input("save graph? (y/n): ")
            
                if save.lower() == "y":

                    oracle_name = str(self.oracle.model.__class__).split('.')[-2]
                    G = nx.from_numpy_array(cf_cand_matrix)

                    for ix, feat in enumerate(result.node_features):
                        # print(feat.mean())
                        G.nodes[ix]["Feature"] = feat.mean()

                    nx.write_gexf(G, f"CFs_figs\\{instance.id}-iRand-{oracle_name}-not_cf.gexf")
                    input(f"Graph saved to CFs_figs\{instance.id}-iRand-{oracle_name}-not_cf.gexf")

            # print(self.chem_flag)
            # input(result)

            if result and self.chem_flag:
                ATOM_MAPS = {
                    "MUTAG": {0: "C", 1: "N", 2: "O", 3: "F", 4: "I", 5: "Cl", 6: "Br"},
                    "Mutagenicity": {0: "C", 1: "N", 2: "O", 3: "F", 4: "I", 5: "Cl", 6: "Br", 7: "S"},
                }
                BOND_MAPS = {
                    1: Chem.BondType.SINGLE,
                    2: Chem.BondType.DOUBLE,
                    3: Chem.BondType.TRIPLE,
                    4: Chem.BondType.AROMATIC
                }

                def nx_to_mol(G: nx.Graph, dataset_name: str, node_label_attr="label", edge_label_attr="label"):
                    atom_map = ATOM_MAPS.get(dataset_name)
                    if atom_map is None:
                        return None  # atom types unknown for this dataset
                    mol = Chem.RWMol()
                    for node, data in G.nodes(data=True):
                        symbol = atom_map.get(data.get(node_label_attr, 0), "C")
                        mol.AddAtom(Chem.Atom(symbol))
                    for u, v, data in G.edges(data=True):
                        if u == v: continue  # skip self-loops
                        bond_type = BOND_MAPS.get(data.get(edge_label_attr, 1), Chem.BondType.SINGLE)
                        mol.AddBond(int(u), int(v), bond_type)
                    try:
                        Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
                        return mol  # valid molecule
                    except Exception:
                        return None  # invalid molecule

                def is_valid_molecule(G: nx.Graph, dataset_name: str, **kwargs) -> bool:
                    return nx_to_mol(G, dataset_name, **kwargs) is not None

                def get_cf_validity(instance, cf_adj: np.ndarray, dataset_name: str) -> bool:
                    """
                    Check if a CF adjacency matrix corresponds to a valid molecule.
                    Requires instance.atom_types to be set (from TUDataset populate()).
                    Returns False if atom types are unavailable for this dataset.
                    """
                    if instance.atom_types is None:
                        return False
                    G = nx.from_numpy_array(cf_adj)
                    for i, atom_idx in enumerate(instance.atom_types):
                        G.nodes[i]['label'] = int(atom_idx)
                    return is_valid_molecule(G, dataset_name)
                
                def plot_molecule(mol, title="", filepath=None, ref_mol=None):
                    # Generate 2D coords
                    import random
                    if ref_mol is not None:
                        try:
                            AllChem.GenerateDepictionMatching2DStructure(mol, ref_mol)
                        except ValueError:
                            AllChem.Compute2DCoords(mol)  # fallback to independent layout
                    else:
                        AllChem.Compute2DCoords(mol)
                    img = Draw.MolToImage(mol, size=(300, 300))

                    fig, ax = plt.subplots()
                    ax.imshow(img)
                    ax.axis("off")
                    ax.set_title(title)
                    if filepath:
                        plt.savefig(filepath, format='pdf', bbox_inches='tight', pad_inches=0, transparent=True)
                        print(f"Saved to {filepath}")
                    plt.show()
                    plt.close()
                
                print(f"result.label: {result.label}")
                chem_valid = get_cf_validity(instance, result.data, "MUTAG")
                print(f"Chemically valid: {chem_valid}")
                cf_adj = result.data
                G = nx.from_numpy_array(cf_adj)
                G_original = nx.from_numpy_array(instance.data)
                if instance.atom_types is not None:
                    for i, atom_idx in enumerate(instance.atom_types):
                        G.nodes[i]['label'] = int(atom_idx)
                        G_original.nodes[i]['label'] = int(atom_idx)
                        
                mol = nx_to_mol(G, "MUTAG")
                mol_original = nx_to_mol(G_original, "MUTAG")
                
                if chem_valid and mol:
                    oracle_name = str(self.oracle.model.__class__).split('.')[-2]
                    # draw_molecule(mol, "MUTAG", filepath=f"CFs_figs\\{self.dataset.name}\\{instance.id}-{explainer_name}-{oracle_name}.png")
                    os.makedirs(f"CFs_figs\\{self.dataset.name}", exist_ok=True)
                    if input("save graph? (y/n): ").lower() == "y":
                        plot_molecule(mol, filepath=f"CFs_figs\\{self.dataset.name}\\{instance.id}-iRand-{oracle_name}-{"cf" if l_input_inst!=l_cf_cand else "not_cf"}.pdf", ref_mol=mol_original)
                        nx.write_gexf(G, f"CFs_figs\\{self.dataset.name}\\{instance.id}-iRand-{oracle_name}-{"cf" if l_input_inst!=l_cf_cand else "not_cf"}.gexf")
                        input(f"Graph saved to CFs_figs\{self.dataset.name}\{instance.id}-iRand-{oracle_name}-{"cf" if l_input_inst!=l_cf_cand else "not_cf"}.gexf")


            return copy.deepcopy(instance)
    

    def real_fit(self):
        pass

    
    def check_configuration(self):
        super().check_configuration()

        if not 'p' in self.local_config['parameters']:
            self.local_config['parameters']['p'] = 0.1

        if not 't' in self.local_config['parameters']:
            self.local_config['parameters']['t'] = 3

    def write(self):
        pass
      
    def read(self):
        pass
    