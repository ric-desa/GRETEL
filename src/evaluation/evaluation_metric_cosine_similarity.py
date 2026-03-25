from src.evaluation.evaluation_metric_base import EvaluationMetric
from src.core.oracle_base import Oracle
from src.core.explainer_base import Explainer
from src.core.embedder_base import Embedder

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import numpy as np


class CosineSimilarityMetric(EvaluationMetric):
    """
    Computes the cosine similarity between the Instance and its counterfactual.
    This is done for all the embedders fitted on the dataset.
    """

    def __init__(self, config_dict=None) -> None:
        super().__init__(config_dict)
        self._name = 'Cosine_Similarity'          

    def evaluate(self, instance_1 , instance_2 , oracle : Oracle=None, explainer : Explainer=None, dataset = None, embedders:dict[Embedder]=None):
        """
        Evaluates the cosine similarity between the embeddings of instance_1 and instance_2.
        
        Parameters:
            instance_1: The original graph.
            instance_2: The counterfactual graph.
            oracle, explainer, dataset: Additional parameters (unused in this snippet).
        
        Returns:
            A dictionary mapping model names to the cosine similarity score between the two embeddings.
        """
        similarity_results = {}

        for embedder in embedders.values():
            try:
                embeddings = embedder.infer([instance_1, instance_2])
            
                # Reshape the embeddings if necessary to be 2D arrays
                emb1 = self.safe_embedding(embeddings[0].reshape(1, -1))
                emb2 = self.safe_embedding(embeddings[1].reshape(1, -1))
                
                # Compute cosine similarity between the two embeddings
                cos_sim = cosine_similarity(emb1, emb2)[0][0]
                similarity_results[embedder.__class__.__name__] = float(cos_sim)
            except Exception:
                similarity_results[embedder.__class__.__name__] = 0.0

        return similarity_results
    
    def safe_embedding(self, emb):
        if np.isnan(emb).any():
            # replace NaNs by 0 (or another neutral value)
            emb = np.nan_to_num(emb, nan=0.0)
        # normalize to unit vector (important for cosine)
        return normalize(emb.reshape(1, -1))