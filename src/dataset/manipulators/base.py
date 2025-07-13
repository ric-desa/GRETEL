import numpy as np

from src.core.configurable import Configurable

class BaseManipulator(Configurable):
    
    def __init__(self, context, local_config, dataset):
        self.dataset = dataset
        super().__init__(context, local_config)
        
    def init(self):
        super().init()
        self.manipulated = False
        self.process()
         
    def process(self):
        for instance in self.dataset.instances:
            node_features_map = self.node_info(instance)
            edge_features_map = self.edge_info(instance)
            graph_features_map = self.graph_info(instance)
            self.manipulate_features_maps((node_features_map, edge_features_map, graph_features_map))
            # overriding the features
            # resize in num_nodes x feature dim
            instance.node_features = self.process_features(instance.node_features, node_features_map, self.dataset.node_features_map)
            instance.edge_features = self.process_features(instance.edge_features, edge_features_map, self.dataset.edge_features_map)
            instance.graph_features = self.process_features(instance.graph_features, graph_features_map, self.dataset.graph_features_map)

           

    def process_instance(self,instance):
        node_features_map = self.node_info(instance)
        edge_features_map = self.edge_info(instance)
        graph_features_map = self.graph_info(instance)
        self.manipulate_features_maps((node_features_map, edge_features_map, graph_features_map))
        # overriding the features
        # resize in num_nodes x feature dim
        instance.node_features = self.process_features(instance.node_features, node_features_map, self.dataset.node_features_map)
        instance.edge_features = self.process_features(instance.edge_features, edge_features_map, self.dataset.edge_features_map)
        instance.graph_features = self.process_features(instance.graph_features, graph_features_map, self.dataset.graph_features_map)

       
    def node_info(self, instance):
        return {}
    
    def graph_info(self, instance):
        return {}
    
    def edge_info(self, instance):
        return {}
    
    def manipulate_features_maps(self, feature_values):
        if not self.manipulated:
            node_features_map, edge_features_map, graph_features_map = feature_values
            self.dataset.node_features_map = self._process_map(node_features_map, self.dataset.node_features_map)
            self.dataset.edge_features_map = self._process_map(edge_features_map, self.dataset.edge_features_map)
            self.dataset.graph_features_map = self._process_map(graph_features_map, self.dataset.graph_features_map)
            self.manipulated = True
    
    def _process_map(self, curr_map, dataset_map):
        _max = max(dataset_map.values()) if dataset_map.values() else -1
        for key in curr_map:
            if key not in dataset_map:
                _max += 1
                dataset_map[key] = _max
        return dataset_map
    
    # def process_features(self, features, curr_map, dataset_map):
    #     if curr_map:
    #         if not isinstance(features, np.ndarray):
    #             features = np.array([])
    #         try:
    #             old_feature_dim = features.shape[1]
    #         except IndexError:
    #             old_feature_dim = 0
    #         # If the feature vector doesn't exist, then
    #         # here we're creating it for the first time
    #         if old_feature_dim:
    #             features = np.pad(features,
    #                             pad_width=((0, 0), (0, len(dataset_map) - old_feature_dim)),
    #                             constant_values=0)
    #         else:
    #             features = np.zeros((len(list(curr_map.values())[0]), len(dataset_map)))
                
    #         for key in curr_map:
    #             print(key, dataset_map[key], curr_map[key])
    #             index = dataset_map[key]
    #             features[:, index] = curr_map[key]
    #             print(features, len(features))
    #             if len(features) == 0:
    #                 features = np.array([np.array([])])
    #             print(features)
                            
    #     return features

    def process_features(self, features, curr_map, dataset_map):
        if not curr_map:
            return features

        D = len(dataset_map)  # total number of feature‑columns

        # 1) Coerce into a (N_old, D_old) numpy array (or zero‐row array)
        if not isinstance(features, np.ndarray) or features.ndim != 2:
            features = np.zeros((0, D), dtype=float)

        # 2) Pad to full width D
        N_old, D_old = features.shape
        if D_old < D:
            pad = ((0, 0), (0, D - D_old))
            features = np.pad(features, pad, constant_values=0.0)

        # 3) Determine row count N for this graph:
        #    either the actual node count, or at least 1
        some_key = next(iter(curr_map))
        actual_N = len(curr_map[some_key])
        N = actual_N if actual_N > 0 else 1

        # 4) If we have zero rows so far, or the row‐count mismatches,
        #    re‑init to exactly (N, D)
        if features.shape[0] != N:
            features = np.zeros((N, D), dtype=float)

        # 5) Fill each column from curr_map; for empty lists it'll just
        #    broadcast a 0.0 into that column
        for key, vals in curr_map.items():
            idx = dataset_map[key]
            col = np.asarray(vals, dtype=float)
            if col.size == 0:
                # empty → fill with zeros
                continue
            if col.shape[0] != N:
                raise ValueError(f"Feature '{key}' has length {col.shape[0]} but expected {N}")
            features[:, idx] = col

        return features