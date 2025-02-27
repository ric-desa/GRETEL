from src.evaluation.evaluation_metric_base import EvaluationMetric
from src.core.oracle_base import Oracle
from src.core.explainer_base import Explainer
from src.evaluation.evaluation_metric_ged import GraphEditDistanceMetric
import torch



class NodeCorrectnessMetric(EvaluationMetric):
    """Verifies that the class from the counterfactual example is different from that of the original instance
    """

    def __init__(self, config_dict=None) -> None:
        super().__init__(config_dict)
        self._name = 'Correctness'
        self._ged = GraphEditDistanceMetric()

    def evaluate(self, instance_1 , instance_2 , oracle : Oracle=None, explainer : Explainer=None, dataset = None):

        label_instance_1 = oracle.predict(instance_1)
        label_instance_2 = oracle.predict(instance_2)
        oracle._call_counter -= 2

        ged = self._ged.evaluate(instance_1, instance_2, oracle)

        result = torch.sum((label_instance_1 != label_instance_2))/len(label_instance_1) if torch.any((label_instance_1 != label_instance_2)) and (ged != 0) else 0
        
        return result