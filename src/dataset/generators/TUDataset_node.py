import torch, numpy as np
from os.path import join,exists
from os import makedirs

from src.dataset.generators.base import Generator
from src.dataset.instances.graph import GraphInstance

from torch_geometric.datasets import TUDataset as downloader

class TUDataset(Generator):

    def prepare_data(self):
        base_path = join(self.context.working_store_path,self.dataset_name)
        self.context.logger.info("Dataset Data Path:\t" + base_path)

        if not exists(base_path):
            self.context.logger.info("Downloading " + self.dataset_name + "...")
            makedirs(base_path, exist_ok=True)
            if self.dataset_name == "DBLP_v1":
                dataset = downloader(base_path, name=self.dataset_name, use_node_attr=False, use_edge_attr=True)
                # Convert one-hot node labels → single integer feature BEFORE saving
                compact = []
                for i in range(len(dataset)):
                    data = dataset[i]   # this gives a Data object
                    # if one-hot was created, reduce it
                    if hasattr(data, 'x') and data.x is not None and data.x.dim() == 2 and data.x.size(1) > 1000:
                        idx = data.x.argmax(dim=1)
                        data.x = idx.unsqueeze(1).to(torch.float32)
                    if hasattr(data, 'node_labels'):
                        del data.node_labels
                    compact.append(data)
                torch.save(compact, join(base_path, f'{self.dataset_name}.pkl'))
                self.context.logger.info(f"Saved dataset {self.dataset_name} in {join(base_path, f'{self.dataset_name}.pkl')}.")
                return base_path
            else:
                dataset = downloader(base_path, name=self.dataset_name, use_node_attr=True, use_edge_attr=True)
            if not exists(join(base_path, f'{self.dataset_name}.pkl')):
                torch.save(dataset, join(base_path, f'{self.dataset_name}.pkl'))
                self.context.logger.info(f"Saved dataset {self.dataset_name} in {join(base_path, f'{self.dataset_name}.pkl')}.")
        return base_path        
       
    
    def init(self):
        self.dataset_name = self.local_config['parameters']['alias']
        base_path = self.prepare_data()
        # read the dataset and process it
        self.read_file = join(base_path, f'{self.dataset_name}.pkl')
        self.generate_dataset()

    def generate_dataset(self):
        if not len(self.dataset.instances):
            self.populate()

    def populate(self):
        print(f"file: {self.read_file}")
        data = torch.load(self.read_file, weights_only=False)
        features_map = {f'attribute_{i}': i for i in range(data[0].x.size(1))} if self.dataset_name not in {"COLLAB", "IMDB-MULTI"} else {f'attribute_{i}': i for i in range(data[0].num_nodes)}
        self.dataset.node_features_map = features_map

        # TODO edge_map, graph_map

        # Collect all node labels of the dataset to build a consistent mapping
        DATASET_NODE_ATTR = {"BZR": 3, "ENZYMES": 18}
        if self.dataset_name == "DBLP_v1":
            # x already contains single integer labels
            num_attr = 0
            all_node_labels = []
            for instance in data:
                node_labels = instance.x.squeeze().long().numpy()
                all_node_labels.extend(node_labels)
        else:
            num_attr = DATASET_NODE_ATTR.get(self.dataset_name, data[0].x.size(1))
            all_node_labels = []
            for instance in data:
                node_labels = instance.x[:, num_attr:].argmax(dim=1).numpy()
                all_node_labels.extend(node_labels)

        unique_labels = sorted(set(all_node_labels))
        global_mapping = {v: i for i, v in enumerate(unique_labels)}
        self.dataset._class_indices = {i: [] for i in range(len(unique_labels))} # Set correct number of classes for dataset
        self.dataset.num_classes = len(unique_labels)
        print(f"Global node label mapping for {self.dataset_name}: {global_mapping}")

        for id, instance in enumerate(data):

            # print(f"labels: {instance.y} -\n x: {instance.x[0]} ")
            # input()
            # print(f"dict: {instance.__dict__}")
            # input()

            if self.dataset_name in {"COLLAB", "IMDB-MULTI", "DBLP_v1"}:
                # adj_matrix = torch.zeros((instance.num_nodes, instance.num_nodes), dtype=torch.float)
                instance.x = torch.zeros((instance.num_nodes, 1))
            adj_matrix = torch.zeros((instance.x.size(0), instance.x.size(0)), dtype=torch.float)
            adj_matrix[instance.edge_index[0], instance.edge_index[1]] = 1.0 

            if adj_matrix.shape[0] == 0:
                print(f"Skipping instance {id} with empty adjacency matrix.")
                continue
            
            edge_features = None
            try:
                edge_features = instance.edge_weights.numpy()
            except AttributeError:
                self.context.logger.info(f'Instance id = {id} does not have edge features.')
            
            node_labels = instance.x[:, num_attr:].argmax(dim=1).numpy()
            node_labels = np.array([global_mapping[v] for v in node_labels], dtype=int)
            # print(f"node labels: {node_labels}")

            self.dataset.instances.append(GraphInstance(id=id, 
                                                        label=node_labels, 
                                                        data=adj_matrix.numpy(),
                                                        graph_features=None,
                                                        node_features=instance.x.numpy(),
                                                        edge_features=edge_features
                                                        ))