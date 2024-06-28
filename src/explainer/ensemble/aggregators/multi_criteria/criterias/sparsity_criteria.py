from src.dataset.instances.graph import GraphInstance
from src.explainer.ensemble.aggregators.multi_criteria.criterias.base_criteria import (
    BaseCriteria,
)
from src.explainer.ensemble.aggregators.multi_criteria.criterias.gain_direction import (
    GainDirection,
)
from src.utils.metrics.sparsity import sparsity_metric


class SparsityCriteria(BaseCriteria[GraphInstance]):
    def gain_direction(self):
        return GainDirection.MINIMIZE

    def calculate(
        self,
        first_instance: GraphInstance,
        second_instance: GraphInstance,
    ) -> float:
        return sparsity_metric(first_instance, second_instance)
