import ecole
import shutil
import os
import sys
import time
import itertools
import numpy as np
import pyscipopt as pyscip
import pickle as pkl
import networkx as nx
from tqdm import trange

from mipgnn.utility import *
from mipgnn import bias_search

from mipgnn.instance import MIPInstance

def generate_instances(name, n_constraint = [500], n_variable = [500], density = [0.2],n_instance = 1000, dataset_directory=os.path.expanduser("~/.cache/mipgnn/datasets"), generator=ecole.instance.SetCoverGenerator):
    path = os.path.join(dataset_directory, name)
    #if os.path.exists(path):
    #    shutil.rmtree(path)
    # create the dataset directory if it does not exist
    os.makedirs(path, exist_ok=True)
    
    for row, col, d in itertools.product(n_constraint, n_variable, density):
        instance_generator = generator(n_rows = row, n_cols = col,density = d) 
        dataset_name = f"set_cover_{row}_{col}_{d}"
        dataset_path = os.path.join(path, dataset_name)
        if os.path.exists(dataset_path):
            shutil.rmtree(dataset_path)
        os.makedirs(dataset_path, exist_ok=True)

        print("Generate with Row:%d,Col:%d,Density:%f" % (row,col,d))
        for n in trange(1,n_instance+1):
            # new instance
            instance = next(instance_generator)
            instance_path = os.path.join(dataset_path, f"{n}.mps")
            instance.write_problem(instance_path)
            # build the bipartite graph
            mip_instance = MIPInstance.load(instance_path)
            G = mip_instance.bipartite_graph()
            graph_path = os.path.join(dataset_path, f"{n}_graph.pkl")
            nx.write_gpickle(G, graph_path)
            # compute biases
            bias_search.search(instance_path, True)



if __name__=="__main__":
    generate_instances("test_dataset", 
            n_constraint=[30], 
            n_variable=[30], 
            density=[0.1]
            )
