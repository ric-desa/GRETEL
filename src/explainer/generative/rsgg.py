import torch

from src.core.factory_base import get_instance_kvargs
from src.explainer.per_cls_explainer import PerClassExplainer

from src.utils.cfg_utils import init_dflts_to_of
from src.utils.samplers.abstract_sampler import Sampler

class RSGG(PerClassExplainer):

    def init(self):
        super().init()
        self.sampler: Sampler = get_instance_kvargs(self.local_config['parameters']['sampler']['class'],
                                                    self.local_config['parameters']['sampler']['parameters'])
        self.sampler.dataset = self.dataset
                
    def explain(self, instance):          
        with torch.no_grad():  
            res = super().explain(instance)

            embedded_features, edge_probs = dict(), dict()
            for key, values in res.items():
                # take the node features and edge probabilities
                embedded_features[key] = values[0]
                edge_probs[key] = values[-1]
                
            cf_instance = self.sampler.sample(instance, self.oracle,
                                              embedded_features=embedded_features,
                                              edge_probabilities=edge_probs)
            
            if cf_instance:
                import networkx as nx, matplotlib.pyplot as plt
                pos = nx.spring_layout(nx.from_numpy_array(instance.data)) # Fix graph orientation
                instance_graph = nx.from_numpy_array(instance.data)
                CF_graph = nx.from_numpy_array(cf_instance.data)
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                nx.draw(instance_graph, pos=pos, ax=axes[0], with_labels=True, cmap='cool', node_color=instance.node_features.mean(axis=1), edge_color='gray')
                axes[0].set_title(f"Initial Graph | Predicted Class: {instance.label}")
                nx.draw(CF_graph, pos=pos, ax=axes[1], with_labels=True, cmap='cool', node_color=cf_instance.node_features.mean(axis=1), edge_color='gray')
                axes[1].set_title(f"Counterfactual Graph | Predicted Class: {cf_instance.label}")
                fig.suptitle(f"True label: {instance.label}")
                plt.show()

                save = input("save graph? (y/n): ")
            
                if save.lower() == "y":

                    oracle_name = str(self.oracle.model.__class__).split('.')[-2]
                    # G = nx.from_numpy_array(instance.data)

                    # for i, feat in enumerate(instance.node_features):
                    #     # print(feat.mean())
                    #     G.nodes[i]["Feature"] = feat.mean()

                    # nx.write_gexf(G, f"C:\\Users\\ACER\Documents\\CS\\Thesis\\Media\\Counterfactual Visualization\\{instance.id}-RSGGCE-original.gexf")

                    G = nx.from_numpy_array(cf_instance.data)

                    for i, feat in enumerate(cf_instance.node_features):
                        # print(feat.mean())
                        G.nodes[i]["Feature"] = feat.mean()

                    nx.write_gexf(G, f"C:\\Users\\ACER\Documents\\CS\\Thesis\\Media\\Counterfactual Visualization\\{instance.id}-RSGGCE-{oracle_name}.gexf")

            
        return cf_instance if cf_instance else instance
    
    def check_configuration(self):
        self.set_proto_kls('src.explainer.generative.gans.graph.model.GAN')
        super().check_configuration()
        #The sampler must be present in any case
        init_dflts_to_of(self.local_config,
                         'sampler',
                         'src.utils.samplers.partial_order_samplers.PositiveAndNegativeEdgeSampler',
                         sampling_iterations=500)