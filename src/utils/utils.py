from collections import OrderedDict
import copy
import json
import os
import torch
import numpy as np

def update_saved_pyg(input_file,output_file):
    old_model =  torch.load(input_file, map_location=torch.device('cpu'))
    fixed_model = OrderedDict([(k.replace("grpah", "graph"), v) if 'grpah' in k else (k, v) for k, v in old_model.items()])
    torch.save(fixed_model,output_file)

def sanitize_dir_pyg(based_dir,prefix,model_name='explainer'):
    for file in os.listdir(based_dir):
        if file.startswith(prefix):            
            model_file_name = os.path.join(based_dir,file,model_name)
            if os.path.exists(model_file_name):
                old_file_name = os.path.join(based_dir,file,"OLD_"+model_name)

                print("Sanitizing: "+model_file_name)

                os.rename(model_file_name, old_file_name)
                print("Renamed to: "+old_file_name)

                update_saved_pyg(old_file_name,model_file_name)
                print("Complete")

def unfold_confs(based_dir,out_dir,prefix,num_folds=10):
    for dir in os.listdir(based_dir):
        if dir.startswith(prefix) and os.path.isdir(os.path.join(based_dir,dir)):
            # os.makedirs(os.path.join(out_dir,dir), exist_ok=True)
            for sub_dir in os.listdir(os.path.join(based_dir,dir)):
                if os.path.isdir(os.path.join(based_dir,dir,sub_dir)):
                    os.makedirs(os.path.join(out_dir,dir,sub_dir), exist_ok=True)
                    print("Processing subfolder: "+os.path.join(based_dir,dir,sub_dir))
                    for conf_file in os.listdir(os.path.join(based_dir,dir,sub_dir)):
                        #print(conf_file)
                        in_file = os.path.join(based_dir,dir,sub_dir,conf_file)
                        out_file = os.path.join(out_dir,dir,sub_dir,conf_file)

                        with open(in_file, 'r') as config_reader:
                            configuration = json.load(config_reader)                                                    
                            for fold_id in range(num_folds):
                                current_conf =  copy.deepcopy(configuration)
                                for exp  in current_conf['explainers']:
                                    exp['parameters']['fold_id']=fold_id
                                
                                out_file = os.path.join(out_dir,dir,sub_dir,conf_file[:-5]+'_'+str(fold_id)+'.json')
                                with open(out_file, 'w') as o_file:
                                    json.dump(current_conf, o_file)
                                print(out_file)
                                
def pad_adj_matrix(adj_matrix, target_dimension):
    # Get the current dimensions of the adjacency matrix
    current_rows, current_cols = adj_matrix.shape
    # Calculate the amount of padding needed for rows and columns
    pad_rows = max(0, target_dimension - current_rows)
    pad_cols = max(0, target_dimension - current_cols)
    # Pad the adjacency matrix with zeros
    return np.pad(adj_matrix, ((0, pad_rows), (0, pad_cols)), mode='constant')

def pad_features(features, target_dimension):
    nodes, feature_dim = features.shape
    if nodes < target_dimension:
        rows_to_add = max(0, target_dimension - nodes)
        to_pad = np.zeros((rows_to_add, feature_dim))
        features = np.vstack([features, to_pad])
    return features

def discretize_to_nearest_integer(tensor: torch.Tensor) -> torch.Tensor:
    """
    Discretize tensor values to the nearest integer.
    
    Args:
        tensor: Input tensor to discretize
        
    Returns:
        Tensor with values rounded to nearest integer
    """
    return torch.round(tensor)

def get_optimizer(cfg, model):
    """
    Create an optimizer for the given model based on configuration.
    
    Args:
        cfg: Configuration dictionary
        model: Model to optimize
        
    Returns:
        Optimizer instance
    """
    optimizer_config = cfg.get('parameters', {}).get('optimizer', {})
    optimizer_type = optimizer_config.get('type', 'Adam').lower()
    lr = optimizer_config.get('lr', 0.01)
    weight_decay = optimizer_config.get('weight_decay', 0.0)
    
    if optimizer_type == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'sgd':
        momentum = optimizer_config.get('momentum', 0.9)
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_type == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        # Default to Adam
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

def build_counterfactual_graph_gc(x, edge_index, graph, oracle, output_actual, device):
    """
    Constructs a counterfactual graph based on the provided edge index, original graph, and results from an oracle model.
    
    Args:
        x (Tensor): The node features tensor.
        edge_index (Tensor): The edge index tensor representing the edges of the counterfactual graph.
        graph (Data): The original graph data object containing node features and other graph-related information.
        oracle (nn.Module): The oracle neural network model used to obtain embeddings and other necessary computations.
        output_actual (Tensor): The actual output tensor from the oracle model, used to determine the counterfactual labels.
        device (str, optional): The device to perform computations on, default is "cuda".
        
    Returns:
        Data: A new Data object representing the counterfactual graph with updated attributes.
    """
    from torch_geometric.data import Data
    
    # Ensure batch is available
    if hasattr(graph, 'batch') and graph.batch is not None:
        batch = graph.batch.to(device)
    else:
        batch = torch.zeros(x.shape[0], dtype=torch.long, device=device)
    
    # Get embedding representation if the oracle supports it
    x_projection = None
    if hasattr(oracle, 'get_embedding_repr'):
        try:
            x_projection = torch.mean(oracle.get_embedding_repr(x, edge_index, batch), dim=0)
        except Exception as e:
            # Fallback if get_embedding_repr fails
            # Use mean of node features as projection
            x_projection = torch.mean(x, dim=0)
    else:
        # Fallback: use mean of node features as projection
        x_projection = torch.mean(x, dim=0)
    
    counterfactual = Data(
        x=x.to(device),
        edge_index=edge_index.to(device),
        y=torch.argmax(output_actual, dim=1).to(device),
        x_projection=x_projection.to(device)
    )
    
    return counterfactual
# from src.utils.utils import update_saved_pyg 

#input_file="/NFSHOME/mprado/CODE/gretel-steel-2/GRETEL/data/explainers/clear_fit_on_tree-cycles_instances-500_nodes_per_inst-28_nodes_in_cycles-7_fold_id=0_batch_size_ratio=0.15_alpha=0.4_lr=0.01_weight_decay=5e-05_epochs=600_dropout=0.1/old_explainer"
#output_file="/NFSHOME/mprado/CODE/gretel-steel-2/GRETEL/data/explainers/clear_fit_on_tree-cycles_instances-500_nodes_per_inst-28_nodes_in_cycles-7_fold_id=0_batch_size_ratio=0.15_alpha=0.4_lr=0.01_weight_decay=5e-05_epochs=600_dropout=0.1/explainer"
#update_saved_pyg(input_file,output_file)


#based_dir='/NFSHOME/mprado/CODE/gretel-steel-2/GRETEL/data/explainers/'
#sanitize_dir_pyg(based_dir,"clear")
#unfold_confs("config/aaai","AAAI/config","ablation")
