from src.evaluation.evaluation_metric_base import EvaluationMetric
from src.core.oracle_base import Oracle
from src.core.explainer_base import Explainer


class FidelityNodesMetric(EvaluationMetric):
    """As correctness measures if the algorithm is producing counterfactuals, but in Fidelity measures how faithful they are to the original problem,
     not just to the problem learned by the oracle. Requires a ground truth in the dataset
    """

    def __init__(self, config_dict=None) -> None:
        super().__init__(config_dict)
        self._name = 'Fidelity_nodes'

    def evaluate(self, instance_1 , instances_2 , oracle : Oracle=None, explainer : Explainer=None, dataset = None):
        r=[]
        for node_id, instance_2 in enumerate(instances_2):
            label_instance_1 = oracle.predict(instance_1)[node_id]
            label_instance_2 = oracle.predict(instance_2)[node_id]
            oracle._call_counter -= 2

            prediction_fidelity = 1 if (label_instance_1 == instance_1.label[node_id]) else 0
            
            counterfactual_fidelity = 1 if (label_instance_2 == instance_1.label[node_id]) else 0

            result = prediction_fidelity - counterfactual_fidelity

            # print("label_instance_1, label_instance_2, instance_1.label, node_id, result", label_instance_1, label_instance_2, instance_1.label, node_id, result)
            # input()
            
            r.append(result)
            
        return r