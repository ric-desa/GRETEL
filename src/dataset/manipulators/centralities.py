import networkx as nx
import numpy as np

from src.dataset.manipulators.base import BaseManipulator


class NodeCentrality(BaseManipulator):
    
    
    def node_info(self, instance):
        #self.context.logger.info("Building centralities for: "+str(instance.id))
        graph = instance._build_nx()
        # Calculate the degree of each node
        degree = dict(graph.degree()) # if len(graph) > 0 else {}
        # Calculate the betweenness centrality
        betweenness_centrality = nx.betweenness_centrality(graph) # if len(graph) > 0 else {}
        # Calculate the closeness centrality
        closeness_centrality = nx.closeness_centrality(graph) # if len(graph) > 0 else {}
        # Calculate the harmonic centrality
        harmonic_centrality = nx.harmonic_centrality(graph) # if len(graph) > 0 else {}
        # Calculate the clustering coefficient
        clustering_coefficient = nx.clustering(graph) # if len(graph) > 0 else {}
        # Calculate the Katz centrality
        katz_centrality = nx.katz_centrality_numpy(graph) # if len(graph) > 0 else {}
        # Calculate the Laplacian centrality
        # print(graph)
        laplacian_centrality = nx.laplacian_spectrum(graph) if len(graph) > 0 else {}
        # print(laplacian_centrality)
        # feature dictionary
        # if graph.number_of_nodes() == 0:
        #     feature_map = {
        #         "degrees":           [0.0],
        #         "betweenness":       [0.0],
        #         "closeness":         [0.0],
        #         "harmonic_centrality":[0.0],
        #         "clustering_coefficient":[0.0],
        #         "katz_centrality":   [0.0],
        #         "laplacian_centrality":[0.0],
        #     }
        # else:
        feature_map = {
            "degrees": list(degree.values()),
            "betweenness": list(betweenness_centrality.values()),
            "closeness": list(closeness_centrality.values()),
            "harmonic_centrality": list(harmonic_centrality.values()),
            "clustering_coefficient": list(clustering_coefficient.values()),
            "katz_centrality": list(katz_centrality.values()),
            "laplacian_centrality": list(laplacian_centrality)
        }
        return feature_map
    
    def __centrality_unconnected_graphs(self, G):
        result = {}
        connected_components = list(nx.connected_components(G))

        for cn in connected_components:
            so_cen = nx.second_order_centrality(G.subgraph(cn))

            for k, v in so_cen.items():
                result[k] = v

        return result
    