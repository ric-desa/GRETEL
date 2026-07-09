import torch, numpy as np

from src.core.factory_base import get_instance_kvargs
from src.explainer.per_cls_explainer import PerClassExplainer

from src.utils.cfg_utils import init_dflts_to_of
from src.utils.samplers.abstract_sampler import Sampler

from rdkit import Chem
from rdkit.Chem import Draw, AllChem
import os
import networkx as nx, matplotlib.pyplot as plt

class RSGG(PerClassExplainer):

    def init(self):
        super().init()
        self.sampler: Sampler = get_instance_kvargs(self.local_config['parameters']['sampler']['class'],
                                                    self.local_config['parameters']['sampler']['parameters'])
        self.sampler.dataset = self.dataset
        self.visualize = self.local_config['parameters'].get('visualize', False)
        self.vis_id = self.local_config['parameters'].get('vis_id', -1)
        self.chem_flag = self.local_config['parameters'].get('chem_flag', False)
                
    def explain(self, instance):
        while True:
            if self.visualize and instance.id != self.vis_id and self.vis_id != -1:
                return instance
            with torch.no_grad():  
                res = super().explain(instance)

                embedded_features, edge_probs = dict(), dict()
                for key, values in res.items():
                    # take the node features and edge probabilities
                    embedded_features[key] = values[0]
                    edge_probs[key] = values[-1]
                    
                cf_instance = self.sampler.sample(instance, self.oracle,
                                                embedded_features=embedded_features,
                                                edge_probabilities=edge_probs)
                
                if cf_instance and self.visualize:
                    pos = nx.spring_layout(nx.from_numpy_array(instance.data)) # Fix graph orientation
                    instance_graph = nx.from_numpy_array(instance.data)
                    CF_graph = nx.from_numpy_array(cf_instance.data)
                    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                    nx.draw(instance_graph, pos=pos, ax=axes[0], with_labels=True, cmap='cool', node_color=instance.node_features.mean(axis=1), edge_color='gray')
                    axes[0].set_title(f"Initial Graph | Predicted Class: {instance.label}")
                    nx.draw(CF_graph, pos=pos, ax=axes[1], with_labels=True, cmap='cool', node_color=cf_instance.node_features.mean(axis=1), edge_color='gray')
                    axes[1].set_title(f"Counterfactual Graph | Predicted Class: {cf_instance.label}")
                    fig.suptitle(f"True label: {instance.label}")
                    plt.show()

                    save = input("save graph? (y/n): ")
                
                    if save.lower() == "y":

                        oracle_name = str(self.oracle.model.__class__).split('.')[-2]

                        G = nx.from_numpy_array(cf_instance.data)

                        for i, feat in enumerate(cf_instance.node_features):
                            # print(feat.mean())
                            G.nodes[i]["Feature"] = feat.mean()

                        nx.write_gexf(G, f"CFs_figs\\{instance.id}-RSGG-CE-{oracle_name}.gexf")
                        input(f"Graph saved to CFs_figs\{instance.id}-RSGG-CE-{oracle_name}.gexf")

                if cf_instance and self.chem_flag:
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
                        cf_adj_binary = (cf_adj > 0.5).astype(int)
                        G = nx.from_numpy_array(cf_adj_binary)
                        if not nx.is_connected(G):
                            return False
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
                    
                    print(f"result.label: {cf_instance.label} | instance.label: {instance.label}")
                    # print(f"orig_pred: {orig_pred} | cf_pred: {cf_pred}")
                    chem_valid = get_cf_validity(instance, cf_instance.data, "MUTAG")
                    print(f"Chemically valid: {chem_valid}")
                    cf_adj = cf_instance.data
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
                            plot_molecule(mol, filepath=f"CFs_figs\\{self.dataset.name}\\{instance.id}-RSGG-CE-{oracle_name}-cf.pdf", ref_mol=mol_original)
                            nx.write_gexf(G, f"CFs_figs\\{self.dataset.name}\\{instance.id}-RSGG-CE-{oracle_name}-cf.gexf")
                            input(f"Graph saved to CFs_figs\{self.dataset.name}\{instance.id}-RSGG-CE-{oracle_name}-cf.gexf")
                
            if self.visualize and not cf_instance and self.vis_id != -1: 
                # input(f"No CF found for instance {instance.id}")
                continue
            break
        return cf_instance if cf_instance else instance
    
    def check_configuration(self):
        self.set_proto_kls('src.explainer.generative.gans.graph.model.GAN')
        super().check_configuration()
        #The sampler must be present in any case
        init_dflts_to_of(self.local_config,
                         'sampler',
                         'src.utils.samplers.partial_order_samplers.PositiveAndNegativeEdgeSampler',
                         sampling_iterations=500)