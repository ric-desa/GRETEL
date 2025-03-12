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

    def evaluate(self, instance_1 , instance_2 , oracle : Oracle=None, explainer : Explainer=None, dataset = None, embedders:dict[Embedder]=None):
        counterfactual = None
        cf_label = None

        if instance_1.id == self.id:
            counterfactual = explainer.explain(instance_1)
            # visualize_tree_cycle(counterfactual)
            cf_label = counterfactual.label.item()
            counterfactual = self.generate_emb_vector(counterfactual, embedders).tolist()
        
        original_embedding = self.generate_emb_vector(instance_1, embedders).tolist()

        return original_embedding, instance_1.label, counterfactual, cf_label
    
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

