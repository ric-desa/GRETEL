import time
from src.evaluation.evaluation_metric_base import EvaluationMetric
from src.core.oracle_base import Oracle
from src.core.explainer_base import Explainer
from src.dataset.instances.graph import GraphInstance


class RuntimeMetricNodes(EvaluationMetric):

    def __init__(self, config_dict=None) -> None:
        super().__init__(config_dict)
        self._name = 'Runtime_nodes'
        self._special = True

    def evaluate(self, instance_1:GraphInstance, instance_2:GraphInstance, oracle : Oracle=None, explainer : Explainer=None, dataset = None):
        times, cfs = [], []
        for node_id in range(instance_1.node_features.shape[0]): # iterate over nodes
            explainer.set_node_id(node_id)
            start_time = time.time()
            counterfactual = explainer.explain(instance_1)
            end_time = time.time()
            # giving the same id to the counterfactual and the original instance 
            counterfactual.id =instance_1.id
            times.append(end_time - start_time)
            cfs.append(counterfactual)
        
        return times, cfs