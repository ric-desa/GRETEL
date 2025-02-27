from src.core.explainer_base import Explainer
from src.core.factory_base import get_class, get_instance_kvargs
from src.core.trainable_base import Trainable
from src.utils.cfg_utils import  inject_dataset, inject_oracle
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from copy import deepcopy

class ExplainerEnsemble(Explainer, Trainable):
    """The base class for the Explainer Ensemble. It should provide the common logic 
    for integrating multiple explainers and produce unified explanations"""

    def init(self):
        super().init()

        self.explanation_aggregator = get_instance_kvargs(self.local_config['parameters']['aggregator']['class'], 
                                                          {'context':self.context,'local_config': self.local_config['parameters']['aggregator']})
        
        self.base_explainers = [ get_instance_kvargs(exp['class'],
                    {'context':self.context,'local_config':exp}) for exp in self.local_config['parameters']['explainers']]

        self.parallel_processing = self.local_config['parameters'].get('parallel_processing', False)

        self.max_workers = self.local_config['parameters'].get('max_workers', len(self.base_explainers))

        
    @classmethod
    def _call_explain(cls, instance, explainer):
        """
        Takes an instance and an explainer and returns the explanation of that instance produced by that explainer.
        The function is designed to be called in a parallelized workflow
        """
        # This function is designed for executing the base explainers in parallel
        exp = explainer.explain(instance)
        exp.producer = explainer
        return exp


    def explain(self, instance):
        # input_label = self.oracle.predict(instance)

        explanations = []
        if self.parallel_processing:
            instances = [instance for i in range(0, len(self.base_explainers))]
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                explanations = list(executor.map(ExplainerEnsemble._call_explain, 
                                                 instances, 
                                                 self.base_explainers))
        else:
            for explainer in self.base_explainers:
                exp = explainer.explain(instance)
                exp.producer = explainer
                explanations.append(exp)

        result = self.explanation_aggregator.aggregate(explanations)
        result.explainer = self

        return result
    

    def real_fit(self):
        pass

    
    def check_configuration(self):
        super().check_configuration()

        inject_dataset(self.local_config['parameters']['aggregator'], self.dataset)
        inject_oracle(self.local_config['parameters']['aggregator'], self.oracle)

        for exp in self.local_config['parameters']['explainers']:
            exp['parameters']['fold_id'] = self.local_config['parameters']['fold_id']
            # In any case we need to inject oracle and the dataset to the model
            inject_dataset(exp, self.dataset)
            inject_oracle(exp, self.oracle)


    def write(self):
        pass
      
    def read(self):
        pass

    @property
    def name(self):
        alias = get_class( self.local_config['parameters']['aggregator']['class'] ).__name__
        return self.context.get_name(self,alias=alias)