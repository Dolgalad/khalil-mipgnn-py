import json
import numpy as np
import ast

from tqdm import tqdm
import os

"""
param:path of folder
return :data_train_X,data_train_Y,data_validation_X,data_validation_Y,data_test_X,data_test_Y
"""
def load_data(path = "DataSet/"):
    path_set = os.listdir(path)
    dataSet = []
    for filename in tqdm(path_set):
        #read features of constraints 
        cons_file = open(path+filename+"/constraints_features.json","rb")
        data = json.load(cons_file)
        data_cons = np.array(data["values"])
        nb_cons = data_cons.shape[0]
        cons_file.close()
        
        #read features of variables 
        var_file = open(path+filename+"/variables_features.json","rb")
        data = json.load(var_file)
        data_var = np.array(data["values"])
        nb_var = data_var.shape[0]
        var_file.close()
            #compose H
        data_cons = np.hstack((np.zeros((data_cons.shape[0],19)),data_cons))
        data_var = np.hstack((data_var,np.zeros((data_var.shape[0],5))))
        matrix_H = np.vstack((data_var,data_cons))
        
        #read label 
        label_file = open(path+filename+"/label.json","rb")
        data = json.load(label_file)
        bestSol = data["Best_Solution"]
        data_label = np.array(bestSol)
        label_file.close()
        #read edge
        edges_file = open(path+filename+"/edges_features.json","rb")
        data = json.load(edges_file)
        matrix_A = np.identity(nb_cons+nb_var)
        for i in range(len(data["values"])):
            iVar,iCons = data["indices"][1][i],data["indices"][0][i] 
            matrix_A[iVar][nb_var+iCons] = data["values"][i]
            matrix_A[iCons+nb_var][iVar] = data["values"][i]
        edges_file.close()
        dataSet.append({"X":matrix_H,"Y":data_label,"A":matrix_A})
        del data_cons
    return np.array(dataSet)


def dumpEdgeFeatures(filename, edge_features, original_indice):
    l = len(original_indice)
    dict_indice = np.zeros((l))
    for i in range(l):
        dict_indice[int(original_indice[i])] = i

    for i in range(len(edge_features.indices[1])):
        edge_features.indices[1][i] = dict_indice[edge_features.indices[1][i]]

    features = {"values": edge_features.values.tolist(),
                "indices": edge_features.indices.tolist()
                }
    data = json.dumps(features)
    file = open(filename, 'w')
    file.write(data)
    file.close()


def dumpRowFeatures(filename, row_features):
    features = {
        "names": ["bias", "objective_cosine_similarity", "is_tight", "dual_solution_value", "scaled_age"],
        "values": row_features.tolist()
    }
    data = json.dumps(features)
    file = open(filename, 'w')
    file.write(data)
    file.close()


def dumpVariableFeatures(filename, variable_features):
    original_indice = variable_features[:, -1]
    l = len(original_indice)
    dict_indice = np.zeros((l))
    for i in range(l):
        dict_indice[int(original_indice[i])] = i
    for i in range(l):
        variable_features[i][-1] = dict_indice[i]

    value = variable_features[variable_features[:, -1].argsort()]
    features = {
        "names": ["objective", "is_type_binary", "is_type_integer", "is_type_implicit_integer", \
                  "is_type_continuous", "has_lower_bound", "has_upper_bound", "normed_reduced_cost", \
                  "solution_value", "solution_frac", "is_solution_at_lower_bound", "is_solution_at_upper_bound", \
                  "scaled_age", "incumbent_value", "average_incumbent_value", "is_basis_lower", "is_basis_basic", \
                  "is_basis_upper", "is_basis_zero"],
        "values": value[:, :-1].tolist()
    }
    data = json.dumps(features)
    file = open(filename, 'w')
    file.write(data)
    file.close()


def dumpSolution_Ecole(filename, pyscip):
    bestsol = pyscip.getBestSol()
    solutionPool = []
    for sol in pyscip.getSols():
        solutionPool.append(np.around(list(ast.literal_eval(sol.__repr__()).values()), 0).astype(int).tolist())
    solutions = {
        "Best_Solution": np.around(list(ast.literal_eval(bestsol.__repr__()).values()), 0).astype(int).tolist(),
        "Solution_Pool": solutionPool
    }
    data = json.dumps(solutions)
    file = open(filename, 'w')
    file.write(data)
    file.close()

# def dumpSolution_Gurobi(filename,solver,nbSolMax = 10,poolGap=0.1):
#     # Limit how many solutions to collect
#     model.setParam(GRB.Param.PoolSolutions, nbSolMax)
#
#     # Limit the search space by setting a gap for the worst possible solution
#     # that will be accepted
#     model.setParam(GRB.Param.PoolGap, poolGap)
#
#     # do a systematic search for the k- best solutions
#     model.setParam(GRB.Param.PoolSearchMode, 2)
#
#     model.optimize()
#     status = model.Status
#     if status in (GRB.INF_OR_UNBD, GRB.INFEASIBLE, GRB.UNBOUNDED):
#         print('The model cannot be solved because it is infeasible or unbounded')
#         sys.exit(1)
#     if status != GRB.OPTIMAL:
#         print('Optimization was stopped with status ' + str(status))
#         sys.exit(1)
# the first solution is the best
#     #get the beat solution
#     bestSol = {}
#     for var in model.getVars():
#         bestSol[var.varName] = var.X

#     #get the solution pool
#     poolSol = {}
#     nSolution = model.SolCount
#     for e in range (nSolution):
#         sol = {}
#         model.setParam(GRB.Param.SolutionNumber, e)
#         for var in model.getVars():
#             sol[var.varName] = var.X
#         poolSol[model.PoolObjVal]=sol
#
#     solutions = {
#         "Nombre_Solutions":nSolution,
# #         "Best_Solution":bestSol,
#         "Solution_Pool":poolSol
#     }
#
#     data = json.dumps(solutions)
#     file = open(filename,'w')
#     file.write(data)
#     file.close()

