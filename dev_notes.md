# MIP-GNN python implementation

Python implementation of the method presented in the paper "MIP-GNN: A Data-Driven Framework for Guiding Combinatorial Solvers" by
Khalil et al.

## Pre trained models

Pretrained models are available in the original GitHub repository. They follow the naming convention : 
<model_class>_<problem_class>_<problem_params>_<training_params>

## Submodules and objects

MIPInstance : instance of a MIP, loaded from a .lp or .mps file
  - load(file: FileObject, filename: str = None) : load a MIP instance from a .mps or .lp file
  - dump(file, filename: str = None, format: str = ".mps") : dump MIP instance to .mps or .lp file
  - get_graph() : get the corresponding MIPGraph instance

MIPBGraph(networkx.Graph) : bipartite graph representing a MIP instance
  - node attributes : 
    - "type": in ["variable", "constraint"]
    - "features": list of features of the node

  - load(file: FileType, filename: str = None) : load a MIPGraph object from a file
  - dump(file: FileType, filename: str = None) : dump MIPGraph object to file
  - get_data() : get the MIPData instance corresponding to the graph

MIPData(torch_geometric.data.Data) : object containing features for each node and edge in the MIP Bipartite graph representation of the problem
  - variable_map : maps the networkx nodes to variable node ids
  - constraint_map : maps the networkx nodes to constraint node ids
  - variable_count : number of variables
  - constraint_count : number of constraints
  - variable_features : variable node features
  - constraint_features : constraint node features
  - constraint_rhs : constraint right hand side values
  - objective : objective function
  - variable_index : indexes of the variables nodes
  - constraint_index : indexes of the constraint nodes
  

MIPGenerator : MIP instance generation

- dataset : creating and loading datasets of MIP problems
- training : training a model on a given dataset
- networks : GNN model implementations
- inference : executing the model
