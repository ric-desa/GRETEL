from src.evaluation.evaluation_metric_base import EvaluationMetric
from src.core.oracle_base import Oracle
from src.core.explainer_base import Explainer
from src.core.embedder_base import Embedder

from sklearn.metrics.pairwise import cosine_similarity


class CosineSimilarityNodesMetric(EvaluationMetric):
    """
    Computes the cosine similarity between the Instance and its counterfactual.
    This is done for all the embedders fitted on the dataset.
    """

    def __init__(self, config_dict=None) -> None:
        super().__init__(config_dict)
        self._name = 'Cosine_Similarity_nodes'          

    def evaluate(self, instance_1 , instances_2 , oracle : Oracle=None, explainer : Explainer=None, dataset = None, embedders:dict[Embedder]=None):
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
            results = []
            for instance_2 in instances_2:
                embeddings = embedder.infer([instance_1, instance_2])
                
                # Reshape the embeddings if necessary to be 2D arrays
                emb1 = embeddings[0].reshape(1, -1)
                emb2 = embeddings[1].reshape(1, -1)
                
                # Compute cosine similarity between the two embeddings
                cos_sim = cosine_similarity(emb1, emb2)[0][0]
                
                results.append(float(cos_sim))

            similarity_results[embedder.__class__.__name__] = results

        return similarity_results