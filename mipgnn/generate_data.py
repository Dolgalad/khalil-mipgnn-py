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
import argparse

import ecole

import mipgnn.configuration as config

from mipgnn.utility import *
from mipgnn import bias_search
from mipgnn.instance import MIPInstance

def generate_instances(name, n_constraint=500, n_variable=500, density=0.2, n_instance=1000, dataset_directory=config.dataset_directory, generator=ecole.instance.SetCoverGenerator):
    path = os.path.join(dataset_directory, name)
    if os.path.exists(path):
        ans=input("Overwrite dataset in {path} ?(y/n)")
        if ans=="n":
            return
        else:
            shutil.rmtree(path)
    # create the dataset directory if it does not exist
    os.makedirs(path, exist_ok=True)
    
    instance_generator = generator(n_rows=n_constraint, n_cols=n_variable, density=density) 

    print(f"Generating {n_instance} instances with n_var={n_variable}, n_constr={n_constraint}, density={density}.")
    print(f"Saving in {path}")
    for n in trange(1,n_instance+1):
        # new instance
        instance = next(instance_generator)
        instance_path = os.path.join(path, f"{n}.mps")
        instance.write_problem(instance_path)
        # build the bipartite graph
        mip_instance = MIPInstance.load(instance_path)
        G = mip_instance.bipartite_graph()
        graph_path = os.path.join(path, f"{n}_graph.pkl")
        nx.write_gpickle(G, graph_path)
        # compute biases
        bias_search.search(instance_path, True)



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name", type=str, help="dataset name, files will be save in config.dataset_directory")
    parser.add_argument("-n", "--instances", type=int, help="number of instances (default=1000)", default=1000)
    parser.add_argument("-V", "--variable-count", type=int, help="number of variables (default=100)", default=100)
    parser.add_argument("-c", "--constraint-count", type=int, help="number of constraints (default=100)", default=100)
    parser.add_argument("-d", "--density", type=float, help="graph density (default=.1)", default=.1)
    parser.add_argument("-D", "--output-dir", type=str, help="output directory, default is config.dataset_directory", default=config.dataset_directory)


    args = parser.parse_args()

    generate_instances(args.name, 
            n_constraint=args.constraint_count, 
            n_variable=args.variable_count, 
            density=args.density,
            n_instance=args.instances,
            dataset_directory=args.output_dir
            )
