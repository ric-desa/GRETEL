from karateclub import FeatherGraph
from src.core.embedder_base import Embedder
import numpy 

class FeatherGraphEmbedder(Embedder):
    def init(self):
       # super.__init__(self)
       # self.wl_iterations = self.local_config['parameters']['wl_iterations']
       self.model = FeatherGraph() #TODO insert here parameters
       self.embedding = None

    def real_fit(self):
        graph_list = [graph_instance.get_nx() for graph_instance in self.dataset.instances]
        self.model.fit(graphs=graph_list)
        self.embedding = self.model.get_embedding()

    def infer(self, graphs:list) -> numpy.array:
        graphs = [graph_instance.get_nx() for graph_instance in graphs]
        return self.model.infer(graphs)

    def get_embedding(self, instance):
        return super().get_embedding(instance)
    
    def get_embeddings(self):
        return super().get_embeddings()

    def fit(self):
        return super().fit()

        