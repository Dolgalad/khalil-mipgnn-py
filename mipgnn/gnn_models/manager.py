"""Utility functions for searching, instantiating and creating GNN models
"""

from mipgnn.gnn_models.EdgeConv.mip_bipartite_simple_class import EdgeConvSimpleNet
from mipgnn.gnn_models.EdgeConv.mip_bipartite_class import EdgeConvNet
from mipgnn.gnn_models.Sage.mip_bipartite_simple_class import SageSimpleNet
from mipgnn.gnn_models.Sage.mip_bipartite_class import SageNet
from mipgnn.gnn_models.GIN.mip_bipartite_simple_class import GINSimpleNet
from mipgnn.gnn_models.GIN.mip_bipartite_class import GINNet


DEFAULT_MODEL_ARGS_STR = '{"hidden":64,"aggr":"mean","regression":false,"num_layers":4}'

model_classes = [EdgeConvSimpleNet, EdgeConvNet,
                 SageSimpleNet, SageNet,
                 GINSimpleNet, GINNet
                ]


def list_model_classes():
    """Returns a list of GNN class objects
    """
    return model_classes

def get_model_class_instance(index=None, name=None):
    """Get a model class instance by its name or index in the list `model_classes`
    """
    if isinstance(index,int) and 0<=index and index<len(model_classes):
        return model_classes[index]
    if name:
        class_names = [cls.__name__ for cls in model_classes]
        p = class_names.index(name)
        print(p, name, class_names)
        if p>=0:
            return model_classes[p]
    

def list_trained_models():
    """Returns a list of models available for evaluation. These are files saved in `configuration.models_directory` and are titled
        <model_class_prefix>_<dataset_name>{.json}
    The JSON file contains the models description including arguments needed for its instantiation.
    """
    return []
