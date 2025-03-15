import time
from src.evaluation.evaluation_metric_base import EvaluationMetric
from src.core.oracle_base import Oracle
from src.core.explainer_base import Explainer
from src.dataset.instances.graph import GraphInstance
from src.evaluation.evaluation_metric_oracle_accuracy_nodes import OracleAccuracyNodesMetric


class SupplMetric(EvaluationMetric):

    def __init__(self, config_dict=None) -> None:
        super().__init__(config_dict)
        self._name = 'Runtime_nodes'
        self._special = True
        self.oracle_acc = OracleAccuracyNodesMetric()

    def evaluate(self, instance_1:GraphInstance, instance_2:GraphInstance, oracle : Oracle=None, explainer : Explainer=None, dataset = None):
        instance_2 = [i for i in range(instance_1.node_features.shape[0])]
        r = self.oracle_acc.evaluate(instance_1=instance_1, instances_2=instance_2, oracle=oracle)

        return r, instance_1.label.tolist()