import os
import argparse
from progress.bar import Bar
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

from mipgnn.instance import MIPInstance
from mipgnn.solver import MIPSolver


def evaluate(solver, dataset_path, output_dir, solver_name=None):
    if solver_name is None:
        solver_name = solver.__class__.__name__
    # create the output directory
    os.makedirs(output_dir, exist_ok=True)

    # solve all instances in the directory
    instances = list([os.path.join(args.dataset_path, f) for f in os.listdir(args.dataset_path) if f.endswith(".lp") or f.endswith(".mps")])
    with Bar(f"Solving with {solver_name}: ", max=len(instances)) as bar:
        for instance in instances:
            instance_name = os.path.splitext(os.path.basename(instance))[0]

            # solve the instance
            #mip = MIPInstance.load(instance)
            sol, history = solver.optimize(filename = instance)
            pkl.dump((sol, history), open(os.path.join(output_dir, f"{instance_name}.pkl"), "wb"))
            bar.next()


def load_history(directory):
    print(f"Loading history data from : {directory}")
    history_data = {"best_objective_value": [],
            "mip_relative_gap": [],
            "num_iterations": [],
            "num_nodes": [],
            "num_remaining_nodes": []
            }

    for f in os.listdir(directory):
        sol,hist = pkl.load(open(os.path.join(directory, f), "rb"))
        for k in history_data: 
            if len(hist[k]):
                history_data[k].append(hist[k][-1])
    return history_data

def view_history(directories, output_dir=None, labels=None):
    if labels is None:
        labels = [os.path.basename(d) for d in directories]
    history_data_list = [load_history(d) for d in directories]
    history_data = history_data_list[0]

    #f, axes = plt.subplots(len(history_data), 1)
    for i, k in enumerate(history_data):
        f, axes = plt.subplots(1, 1)

        axes.set_title(k)
        data = [h[k] for h in history_data_list]
        axes.boxplot(data, labels=labels)
        axes.set_xticks(1+np.arange(len(labels)), rotation=35)
        plt.tight_layout()
        if output_dir:
            plt.savefig(os.path.join(output_dir, k+".png"))
        else:
            plt.show()



 

    

if __name__=="__main__":
    from mipgnn.mipgnn import MIPGNNSolver

    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str, help="model to evaluate")
    parser.add_argument("dataset_path", type=str, help="path to test dataset")
    parser.add_argument("--output-dir", type=str, help="output directory (defaults to current directory", default="./output")

    args = parser.parse_args()

    # initialize the solvers
    solvers = [MIPGNNSolver("SimpleNet_set_cover_n100_nv100_nc100_d0.1"),
            MIPGNNSolver("SimpleNet_set_cover_n100_nv100_nc100_d0.1", branching_method="local_branching_exact")
            ]
    solvers +=[MIPSolver()]
   
    solver_names =  ["EdgeConv~lb", "EdgeConv-elb"]
    solver_names += [f"CPLEX"]
    _ = [solver.quiet() for solver in solvers]
    history_directories = []
    #solver.quiet()

    os.makedirs(args.output_dir, exist_ok=True)
    for i,solver in enumerate(solvers):
        output_dir = os.path.join(args.output_dir, str(i))
        history_directories.append(output_dir)
        evaluate(solver, args.dataset_path, output_dir, solver_name=solver_names[i])

    view_history(history_directories, output_dir=args.output_dir, labels=solver_names)

   





    


