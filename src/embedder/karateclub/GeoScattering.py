from karateclub import GeoScattering
from src.core.embedder_base import Embedder
import numpy
import networkx as nx

class GeoScatteringEmbedder(Embedder):
    def init(self):
       self.model = GeoScattering() #TODO insert here parameters
       self.embedding = None

    def real_fit(self):
        graph_list = [graph_instance.get_nx() for graph_instance in self.dataset.instances]
        graph_list = [G if G.number_of_nodes() > 0 else nx.empty_graph(1) for G in graph_list]
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
