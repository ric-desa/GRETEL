from karateclub import FeatherGraph
from src.core.embedder_base import Embedder
import numpy as np
import networkx as nx

class FeatherGraphEmbedder(Embedder):
    def init(self):
       # super.__init__(self)
       # self.wl_iterations = self.local_config['parameters']['wl_iterations']
       self.model = FeatherGraph() #TODO insert here parameters
       self.embedding = None

    def real_fit(self):
        graph_list = [graph_instance.get_nx() for graph_instance in self.dataset.instances]
        graph_list = [G if G.number_of_nodes() > 0 else nx.empty_graph(1) for G in graph_list]
        self.model.fit(graphs=graph_list)
        self.embedding = self.model.get_embedding()

        # raw = self.model.get_embedding()
        # D   = len(raw[0])
        # idx = 0
        # graph_embeds = []
        # for G in graph_list:
        #     n = G.number_of_nodes()
        #     if n == 0:
        #         graph_embeds.append(np.zeros(D))
        #     else:
        #         block = np.stack(raw[idx:idx+n])
        #         idx += n
        #         graph_embeds.append(self.pool_fn(block))
        # self.embedding = np.vstack(graph_embeds)  # ← replace the plain np.array(...) call

    def infer(self, graphs:list) -> np.array:
        graphs = [graph_instance.get_nx() for graph_instance in graphs]
        return self.model.infer(graphs)

    def get_embedding(self, instance):
        return super().get_embedding(instance)
    
    def get_embeddings(self):
        return super().get_embeddings()

    def fit(self):
        return super().fit()

        