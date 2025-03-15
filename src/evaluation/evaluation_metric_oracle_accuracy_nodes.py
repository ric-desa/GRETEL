from src.evaluation.evaluation_metric_base import EvaluationMetric
from src.core.oracle_base import Oracle
from src.core.explainer_base import Explainer


class OracleAccuracyNodesMetric(EvaluationMetric):
    """As correctness measures if the algorithm is producing counterfactuals, but in Fidelity measures how faithful they are to the original problem,
     not just to the problem learned by the oracle. Requires a ground truth in the dataset
    """

    def __init__(self, config_dict=None) -> None:
        super().__init__(config_dict)
        self._name = 'Oracle_Accuracy_nodes'

    def evaluate(self, instance_1 , instances_2 , oracle : Oracle=None, explainer : Explainer=None, dataset = None):
        r = []
        for node_id, instance_2 in enumerate(instances_2):
            predicted_label_instance_1 = oracle.predict(instance_1)[node_id]
            oracle._call_counter -= 1
            real_label_instance_1 = instance_1.label[node_id]

            result = 1 if (predicted_label_instance_1 == real_label_instance_1) else 0
            r.append(result)
        
        return r