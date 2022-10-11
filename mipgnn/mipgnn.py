import time
import json
import os
import tempfile

import torch
import numpy as np

from mipgnn.instance import MIPInstance
from mipgnn.solver import MIPSolver

from mipgnn import inference
from mipgnn import predict
from mipgnn import callbacks_cplex
from mipgnn.gnn_models import manager as gnn_manager

class MIPGNNSolver(MIPSolver):
    def __init__(self, model_name, **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.branching_method = kwargs.get("branching_method", "local_branching_approx")
        self.branching_priorities = kwargs.get("branching_priorities", False)
        self.branching_direction = kwargs.get("branching_direction", 1)
        self.node_selection = kwargs.get("node_selection", False)

        self.branch_cb = None
        self.node_cb = None

    def update_history(self):
        """update the history dictionary
        """
        super().update_history()
    
    def get_vcgraph(self, instance):
        """Load the variable to constraint graph if the file exists or create the graph
        """
        return instance.vcgraph()
   
    def get_model(self, model_name, model_class, model_args):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if isinstance(model_args, str):
            model_args = json.loads(model_args)
        model_cls = gnn_manager.get_model_class_instance(name=model_class)
        #print("MOdel class", model_cls.__name__)
        #model = SimpleNet(**model_args)
        model = model_cls(**model_args)
        model_state_path = os.path.expanduser(f"~/.cache/mipgnn/models/{model_name}")
        model.load_state_dict(torch.load(model_state_path, map_location=device))
        return model

    def get_prediction(self, model_name, graph):
        """Use the model to predict variable biases
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model_info = {}
        model_info_path = os.path.expanduser(f"~/.cache/mipgnn/models/{model_name}.json")
        with open(model_info_path, "r") as f:
            model_info = json.loads(f.read())

        model = self.get_model(model_name, model_info["model_class"], model_info["model_args"])

        data, node_to_varnode, _ = predict.create_data_object(graph, 0.0)
        #print("DATAAAAAAAAAAAAAA")
        #print(data)
        #print(data.var_node_features)
        #print()
        model.eval()
    
        data = data.to(device)
    
        # out = model(data).max(dim=1)[1].cpu().detach().numpy()
        prediction = model(data).exp()[:,1].cpu().detach().numpy()
    
        #y_real = data.y_real.cpu().detach().numpy()
        # return out, node_var, var_node
        #return out, node_to_varnode#, y_real

        #prediction, node_to_varnode = predict.get_prediction(model_name=model, graph=graph)
        dict_varname_seqid = predict.get_variable_cpxid(graph, node_to_varnode, prediction)
        return prediction, node_to_varnode, dict_varname_seqid

    def get_local_branching_coefficients(self, prediction, lb=.1, ub=.9):
        pred_one_coeff = (prediction >= ub) * (-1)
        pred_zero_coeff = (prediction <= lb)
        num_ones = -np.sum(pred_one_coeff)
        coeffs = pred_one_coeff + pred_zero_coeff

        local_branching_coeffs = [list(range(len(prediction))), coeffs.tolist()]
        return local_branching_coeffs, num_ones
    def set_node_selection_callbacks(self, prediction, zero_damping=1.0, freq_best=100):
        # score variables based on bias prediction
        scores = np.max(((1-prediction), prediction), axis=0)
        rounding = np.round(prediction)
        
        #print(np.mean(scores), np.mean(rounding))
        #print(np.argsort(prediction), np.sort(prediction)[:10], np.sort(prediction)[-10:])

        self.branch_cb = self.register_callback(callbacks_cplex.branch_attach_data2)
        self.node_cb = self.register_callback(callbacks_cplex.node_selection3)

        self.branch_cb.scoring_function = 'sum' #'estimate'
        self.branch_cb.scores = scores
        self.branch_cb.rounding = rounding
        self.branch_cb.zero_damping = zero_damping

        self.node_cb.last_best = 0
        self.node_cb.freq_best = freq_best

        node_priority = []
        self.branch_cb.node_priority = node_priority
        self.node_cb.node_priority = node_priority            

        self.branch_cb.time = 0
        self.node_cb.time = 0



    def optimize(self, filename = None, lb_threshold=5):
        #print("FILENAME        :" ,filename)
        if filename:
            instance = MIPInstance.load(filename)
        self.read(instance.filename)

        t_start = self.get_time()
    
        t_vcg = time.time()
        vcg = self.get_vcgraph(instance)
        t_vcg = time.time() - t_vcg
        #print("BCG", vcg, t_vcg)
        #print("INSTAcn" , instance)
    
        t_pred = time.time()
        #prediction, node_to_varnode = self.get_prediction(model_name, graph)
        #prediction, node_to_varnode = predict.get_prediction(model_name=model, graph=graph)
        #dict_varname_seqid = predict.get_variable_cpxid(graph, node_to_varnode, prediction)
        prediction, node_to_varnode, dict_varname_seqid = self.get_prediction(self.model_name, vcg)
    
        var_names = list(self.variables.get_names())
        prediction_reord = [dict_varname_seqid[var_name][1] for var_name in var_names]
        prediction = np.array(prediction_reord)
        t_pred = time.time() - t_pred
    
        local_branching_coeffs, num_ones = self.get_local_branching_coefficients(prediction)
    
        if self.branching_method == "local_branching_approx":
            self.linear_constraints.add(
                lin_expr=[local_branching_coeffs],
                senses=['L'],
                rhs=[float(lb_threshold - num_ones)],
                names=['local_branching'])
        
        elif self.branching_method == 'local_branching_exact':
            self.branch_cb = self.register_callback(callbacks_cplex.branch_local_exact)
    
            self.branch_cb.coeffs = local_branching_coeffs
            self.branch_cb.threshold = lb_threshold - num_ones
            self.branch_cb.is_root = True

        if self.branching_priorities:
            inference.set_cplex_priorities(self, prediction, self.branching_direction)

        if self.node_selection:
            self.set_node_selection_callbacks(prediction)


        timelimit = self.solver_parameters.get("timelimit", 60)
        #print("Timelimit : ", timelimit)
        t_remaining_cplex = timelimit - self.get_time() + t_start #self.solver_parameters.get("timelimit", t_pred)
        #print("Time used : ", self.get_time() - t_start)
        #print("Remaining time : ", t_remaining_cplex)
        #print("Good rem : ", timelimit - self.get_time() + t_start)
        self.parameters.timelimit.set(t_remaining_cplex)

        solution = None
        
        if t_remaining_cplex > 0:
            try:
                self.parameters.timelimit.set(t_remaining_cplex)
                t0 = time.time()
                self.solve()
                t0 = time.time() - t0
                self.history["t_solve"] = t0
                solution = self.solution.get_values()
            except:
                self.history["t_solve"] = t_remaining_cplex

        #print(self.solution.is_primal_feasible())
    
        t_end = self.get_time()
        
        self.history["t_start"]= t_start
        self.history["t_vcg"] = t_vcg
        self.history["t_pred"] = t_pred
        self.history["time_end"]= t_end
        self.history["solution.is_primal_feasible"] = self.solution.is_primal_feasible()
        print(self.history)
        print(self.history["t_vcg"] + self.history["t_pred"] + self.history["t_solve"])
        
        #print("\nsolution cost        : ", np.dot(self.solution.get_values(), instance.c))
        #print("best objective value : ", self.info_callback.history["best_objective_value"][-1])
        #d = abs(np.dot(self.solution.get_values(), instance.c) - self.info_callback.history["best_objective_value"][-1])
        #print("difference           : ",d) 

    
        return solution, {**self.history, **self.info_callback.history}



if __name__=="__main__":
    # initialize the solver
    solver = MIPGNNSolver("SimpleNet_set_cover_n100_nv100_nc100_d0.1")
    #solver.quiet()

    # path to MIP instance
    mip_path = "/home/aschulz/.cache/mipgnn/datasets/set_cover_n1000_nv1000_nc1000_d0.1/test/15.mps"
    #import cplex
    #cpx = cplex.Cplex(mip_path)
    #exit()


    # load a problem
    mip = MIPInstance.load("/home/aschulz/.cache/mipgnn/datasets/set_cover_n1000_nv1000_nc1000_d0.1/test/15.mps")

    # solve the problem
    r, h = solver.optimize(mip)
    
    print(solver.info_callback.history)

    import matplotlib.pyplot as plt

    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(solver.info_callback.history["time"], solver.info_callback.history["best_objective_value"] )
    plt.plot(solver.info_callback.history["time"], solver.info_callback.history["incumbent_objective_value"] )
    plt.subplot(3,1,2)
    plt.plot(solver.info_callback.history["time"], solver.info_callback.history["mip_relative_gap"])
    plt.subplot(3,1,3)
    plt.plot(solver.info_callback.history["time"], solver.info_callback.history["num_nodes"])

    plt.show()
