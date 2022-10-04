from mipgnn.gnn_models.EdgeConv.mip_bipartite_class import SimpleNet as EdgeConv
from mipgnn.gnn_models.EdgeConv.mip_bipartite_simple_class import SimpleNet as EdgeConvSimple

from mipgnn.gnn_models.GIN.mip_bipartite_class import SimpleNet as GIN
from mipgnn.gnn_models.GIN.mip_bipartite_simple_class import SimpleNet as GINSimple

from mipgnn.gnn_models.Sage.mip_bipartite_class import SimpleNet as Sage
from mipgnn.gnn_models.Sage.mip_bipartite_simple_class import SimpleNet as SageSimple


def list_gnn_models():
    gnn_models = [("EdgeConv", EdgeConv),
        ("EdgeConvSimple", EdgeConvSimple),
        ("GIN", GIN),
        ("GINSimple", GINSimple),
        ("Sage", Sage),
        ("SageSimple", SageSimple)
        ]
    return gnn_models
