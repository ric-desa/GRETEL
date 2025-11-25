import time
from src.evaluation.evaluation_metric_base import EvaluationMetric
from src.core.oracle_base import Oracle
from src.core.explainer_base import Explainer
from src.core.embedder_base import Embedder
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class EmbedDatasetMetric(EvaluationMetric):

    def __init__(self, config_dict=None) -> None:
        super().__init__(config_dict)
        self._name = 'EmbedDataset'
        self._special = True
        self.id = config_dict['parameters']['id']
        self.stop_after_n_found = config_dict['parameters']['stop_after_n_found']
        self.stop_explaining = False
        self.classes_to_explain = config_dict['parameters']['classes_to_explain'] # e.g. Tree, Cycle
        self.classes_left =  {clss:self.stop_after_n_found for clss in self.classes_to_explain}

    def evaluate(self, instance_1 , instance_2 , oracle : Oracle=None, explainer : Explainer=None, dataset = None, embedders:dict[Embedder]=None):
        counterfactual = None
        counterfactual_emb = None
        cf_label = None

        # find CFs if a specific index is given (default -1, see config\snippets\embed_metrics.json) or find CFs for instances with labels in classes_to_explain
        find_cf = True if ((not self.stop_explaining) and ((self.id>=0 and instance_1.id == self.id) or (instance_1.label in self.classes_to_explain and self.classes_left[instance_1.label]!=0))) else False 
        # instance_1.label == 0 : # e.g. find CF only for trees (0) or cycles (1)
        
        if find_cf:
            counterfactual = explainer.explain(instance_1)
            # visualize_tree_cycle(counterfactual)
            cf_lbl = counterfactual.label
            cf_label = cf_lbl if isinstance(cf_lbl, int) else counterfactual.label.item()
            # print(f"Finding CF for class {instance_1.label} --> {cf_label}");input('press to continue…')
            counterfactual_emb = self.generate_emb_vector(counterfactual, embedders).tolist()
            if cf_lbl!=instance_1.label:    
                self.classes_left[instance_1.label] -= 1    # decrement the instances left to explain for the specific class
                if all(self.classes_left[clss]==0 for clss in self.classes_to_explain):  
                    self.stop_explaining = True                 # stop after n CF found for each class (ignore if negative)
                    print(f"{self.stop_after_n_found} CFs found for classes in {self.classes_to_explain} | Explaining stopped")
        
        original_embedding = self.generate_emb_vector(instance_1, embedders).tolist()

        return original_embedding, instance_1.label, counterfactual_emb, cf_label, counterfactual
    
    def generate_emb_vector(self, inst, embedders):
        embeddings_list = []
        for embedder in embedders.values():
            emb = embedder.infer([inst])[0].reshape(1, -1)
            embeddings_list.append(emb)
        # Concatenate all embeddings horizontally to form a fixed-size vector per instance
        embedding = np.concatenate(embeddings_list, axis=1) if embeddings_list else np.array([])
        
        return embedding
    
def visualize_tree_cycle(G):
    G = nx.from_numpy_array(G.data)
    pos = nx.spring_layout(G) 
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=700, font_size=10)
    plt.title("Generated Tree-Cycle Structure")
    plt.show()

