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

#from mipgnn.utility import *
from mipgnn import bias_search
from mipgnn import cplex_utils
from mipgnn.instance import MIPInstance

def generate_instances(name, n_constraint=500, n_variable=500, density=0.2, n_instance=1000, dataset_directory=config.dataset_directory, generator=ecole.instance.SetCoverGenerator, train_test_split=0.0):
    path = os.path.join(dataset_directory, name)
    if os.path.exists(path):
        ans=input(f"Overwrite dataset in {path} ? (yes/no/continue) ")
        if ans=="n" or ans=="no":
            return
        elif ans=="c" or ans=="continue":
            pass
        else:
            shutil.rmtree(path)
    # create the dataset directory if it does not exist
    os.makedirs(path, exist_ok=True)
    
    instance_generator = generator(n_rows=n_constraint, n_cols=n_variable, density=density) 

    print(f"Generating {n_instance} instances with n_var={n_variable}, n_constr={n_constraint}, density={density}.")
    print(f"Saving in {path}")
    
    pbar = trange(1, n_instance+1)
    for n in pbar:
        # new instance
        pbar.set_description("generating instance ")

        instance_path = os.path.join(path, f"{n}.mps")
        if not os.path.exists(instance_path):
            instance = next(instance_generator)
            instance.write_problem(instance_path)
        # build the bipartite graph
        graph_path = os.path.join(path, f"{n}_graph.pkl")
        if not os.path.exists(graph_path):
            mip_instance = MIPInstance.load(instance_path)
            G = mip_instance.bipartite_graph()
            nx.write_gpickle(G, graph_path)
        # compute biases
        bias_path = os.path.join(path, f"{n}_graph_bias.pkl")
        pool_path = os.path.join(path, f"{n}_pool.npz")
        if not os.path.exists(bias_path) or not os.path.exists(pool_path):
            pbar.set_description("bias search         ")
            bias_search.search(instance_path, True, threads=10)

    # train test split
    if train_test_split != 0.0:
        idx = np.arange(n_instance) + 1
        test_idx = np.random.choice(idx, int(train_test_split * n_instance))
        test_path = os.path.join(path, "test")
        train_path = os.path.join(path, "train")
        os.makedirs(test_path, exist_ok=True)
        os.makedirs(train_path, exist_ok=True)
        for i in idx:
            if i in test_idx:
                os.system(f"mv {path}/{i}.mps {path}/{i}_* {test_path}/")
            else:
                os.system(f"mv {path}/{i}.mps {path}/{i}_* {train_path}/")


def is_instance_file(path):
    """Check if path corresponds to a MILP problem file
    """
    return path.endswith(".mps") or path.endswith(".lp")
def get_instance_list(path):
    """List MILP instance files in a directory
    """
    if not os.path.exists(path):
        return []
    return [f for f in os.listdir(path) if is_instance_file(f)]

def save_instance_as_mps(original_path, new_path):
    """Save an instance as .mps file
    """
    cpx = cplex_utils.get_silent_cpx(instance_path=original_path)
    cpx.write(new_path)



def process_instance_collection(path, keep_originals=False):
    """Process a collection of MILP instances, renaming them, creating VariableConstraintGraph objects, storing solution pool, etc...
    """
    instance_list = get_instance_list(path)
    n_instance = len(instance_list)
    pbar = trange(1, n_instance+1)
    for n in pbar:
        instance = instance_list[n-1]
        instance_path = os.path.join(path, instance)
        # rename and save as .mps file
        if not instance.endswith(".mps"):
            new_instance_path = f"{path}/{n}.mps"
            save_instance_as_mps(instance_path, new_instance_path)
            if not keep_originals:
                os.remove(instance_path)
            instance_path = new_instance_path
        # build the bipartite graph
        graph_path = os.path.join(path, f"{n}_graph.pkl")
        if not os.path.exists(graph_path):
            mip_instance = MIPInstance.load(instance_path)
            G = mip_instance.bipartite_graph()
            nx.write_gpickle(G, graph_path)
        # compute biases
        bias_path = os.path.join(path, f"{n}_graph_bias.pkl")
        pool_path = os.path.join(path, f"{n}_pool.npz")
        if not os.path.exists(bias_path) or not os.path.exists(pool_path):
            pbar.set_description("bias search         ")
            bias_search.search(instance_path, True, threads=10)




if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, help="dataset name, files will be save in config.dataset_directory", default="")
    parser.add_argument("-n", "--instances", type=int, help="number of instances (default=1000)", default=1000)
    parser.add_argument("-V", "--variables", type=int, help="number of variables (default=100)", default=100)
    parser.add_argument("-c", "--constraints", type=int, help="number of constraints (default=100)", default=100)
    parser.add_argument("-d", "--density", type=float, help="graph density (default=.1)", default=.1)
    parser.add_argument("-D", "--output-dir", type=str, help="output directory, default is config.dataset_directory", default=config.dataset_directory)
    parser.add_argument("-s", "--train-test-split", type=float, help="split into training and testing sets", default=0.0)
    parser.add_argument("--process-collection", type=str, help="process collection of instances", default="")


    args = parser.parse_args()

    if len(args.name)==0:
        args.name = f"set_cover_n{args.instances}_nv{args.variables}_nc{args.constraints}_d{args.density}"

    # process collection
    if os.path.exists(args.process_collection):
        process_instance_collection(args.process_collection)
        exit()
    generate_instances(args.name, 
            n_constraint=args.constraints, 
            n_variable=args.variables, 
            density=args.density,
            n_instance=args.instances,
            dataset_directory=args.output_dir,
            train_test_split=args.train_test_split,
            )
