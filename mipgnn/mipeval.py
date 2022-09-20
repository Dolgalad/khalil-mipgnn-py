import submitit
import glob
import os
from cplex.exceptions import CplexError
from pathlib import Path
import time
import math

from mipgnn import inference


def combine_jobs(dict_list):
    for counter, dict_cur in enumerate(dict_list):
        print("job %d/%d" % (counter+1, len(dict_list)))
        #inference.mipeval(**dict_cur)
        try:
            inference.mipeval(**dict_cur)
        except CplexError as exc:
            print("errjob %d/%d" % (counter+1, len(dict_list)))
            print(exc)
            continue
        except Exception as e:
            print("unexpected error")
            continue


def chunks(lst, n):
   """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


num_cpus = 1
mem_gb = 8
timelimit = 1800
memlimit = int(mem_gb/2.0)*1024

problem_class = "gisp" #"fcmnf/L_n200_p0.02_c500" #"gisp"
data_main_path = os.path.expanduser("~/.cache/mipgnn/datasets/test_dataset/set_cover_30_30_0.1/") #"../datagen/data/"
model_name = "SG"
models_path = os.path.expanduser("~/.cache/mipgnn/models/") #"../gnn_models/model_new/%s" % (model_name)
model_hyperparams = "train0.0"
# data_specific_path = "%s/p_hat300-2.clq/mipeval/" % (problem_class.replace('/','_'))
# path_prefix = "%s/%s" % (data_main_path, data_specific_path)
# mps_paths = [str(path) for path in Path(path_prefix).rglob('*.mps')]
output_dir = "OUTPUT_new3/"

barebones = 0
configs = {}
#configs['default_3h8t2e'] = {'method':['default'], 'barebones':0, 'cpx_emphasis':2, 'cpx_threads':num_cpus}
#configs['default_emptycb-0'] = {'method':['default_emptycb'], 'barebones':barebones}
#configs['default-%d' % (barebones)] = {'method':['default'], 'barebones':barebones}
#configs['node_selection-%d-100' % (barebones)] = {'method':['node_selection'], 'barebones':barebones, 'freq_best':100}
#configs['primal_mipstart-%d' % (barebones)] = {'method':['primal_mipstart'], 'barebones':barebones, 'mipstart_strategy':'repair'}
configs['branching_priorities-%d' % (barebones)] = {'method':['branching_priorities'], 'barebones':barebones}
#configs['combined-%d-agg' % (barebones)] = {'method':['primal_mipstart', 'node_selection', 'branching_priorities'], 'barebones':barebones, 'freq_best':100, 'num_mipstarts':10}
#configs['node_selection-%d-100-%s_%s' % (barebones, model_name,  model_hyperparams)] = {'method':['node_selection'], 'barebones':barebones, 'freq_best':100}

dict_list = []
graphs_path = data_main_path #'/home/khalile2/projects/def-khalile2/software/DiscreteNet/discretenet/problems/gisp/graphs'
graphs_filenames = [os.path.basename(graph_fullpath) for graph_fullpath in glob.glob(graphs_path + "/*.clq")] 
graphs_filenames = [f for f in os.listdir(graphs_path) if f.endswith("graph.pkl")]
#graphs_filenames = ["keller4.clq", "p_hat300-1.clq", "gen200_p0.9_44.clq"] #"C250.9.clq"]

#for graph in graphs_filenames:
#    data_specific_path = "%s/%s/mipeval/" % (problem_class.replace('/','_'), graph)
#    path_prefix = data_main_path #"%s/%s" % (data_main_path, data_specific_path)
#
#    model_prefix = models_path if 'C250.9.clq' not in graph else models_path + '_gisp'
#    
#    graph_trained = graph #'C125.9.clq'
#    
#    print("PATH prefix ", path_prefix)
#    print("Graph ", graph)
#    for mps_path in Path(path_prefix).glob('*.mps'):
#        #print(mps_path)
#        instance_path_noext = os.path.splitext(mps_path)[0]
#        #vcg_path = "%s_graph_bias.pkl" % instance_path_noext
#        instance_params_path = "%s_parameters.pkl" % instance_path_noext
#        for config_name, config_dict in configs.items():
#            config_dict_final = dict(config_dict)
#            config_dict_final['timelimit'] = timelimit
#            config_dict_final['memlimit'] = memlimit
#
#            if 'default' not in config_dict['method']:
#                model_path = os.path.join(models_path, "EC_set_cover_30_30_0.1_0.0_0") #'%s_%s_%s' % (model_prefix, graph_trained, model_hyperparams)
#                #if not Path(instance_params_path).is_file() or not Path(model_path).is_file():
#                #    print('Failed for %s' % mps_path)
#                #    print(model_path)
#                #    print(config_dict)
#                #    continue
#                config_dict_final['model'] = model_path
#                config_dict_final['instance_params'] = instance_params_path
#                #config_dict_final['graph'] = vcg_path
#
#            # instance, logfile
#            instance_name_noext = os.path.splitext(os.path.basename(mps_path))[0]
#            config_dict_final['instance'] = str(mps_path)
#            config_dict_final['cpx_tmp'] = '/scratch/khalile2/cpx_tmp'
#            config_dict_final['logfile'] = '%s/%s/%s/' % (output_dir, data_specific_path, config_name)#, instance_name_noext)
#            os.makedirs(config_dict_final['logfile'], exist_ok=True)
#            config_dict_final['logfile'] = '%s/%s.out' % (config_dict_final['logfile'], instance_name_noext)
#            #config_dict_final['logfile'] = 'sys.stdout'
#            dict_list += [config_dict_final]

#combine_jobs([dict_list[1]])
#exit()


models_dir = os.path.expanduser("~/.cache/mipgnn/models")
model_names = []
model_speedups = []
cplex_gaps = []
mipgnn_gaps = []
for model in os.listdir(models_dir):
    if model.endswith(".log") or os.path.isdir(os.path.join(models_dir, model)):
        continue
    print("MIOOO : , " , model)

    c = 0
    model_names.append(model)
    instance_dir = os.path.expanduser("~/.cache/mipgnn/datasets/test_dataset/set_cover_500_500_0.1/")
    instances = [f for f in os.listdir(instance_dir) if f.endswith(".mps")]
    speedups = []
    cpx_gaps = []
    gnn_gaps = []
    for instance in instances:
        instance_name = instance.replace(".mps", "")
        params = {"method": "branching_priorities",
            "instance": os.path.expanduser(f"~/.cache/mipgnn/datasets/test_dataset/set_cover_500_500_0.1/{instance_name}.mps"),
            "graph": os.path.expanduser(f"~/.cache/mipgnn/datasets/test_dataset/set_cover_500_500_0.1/{instance_name}_graph_bias.pkl"),
            "model": os.path.expanduser(f"~/.cache/mipgnn/models/{model}"),
            "timelimit": 5,
            "logfile": None,
            }
    
        su, mgap, cgap = inference.mipeval(**params)
        speedups.append(su)
        cpx_gaps.append(cgap)
        gnn_gaps.append(mgap)
        c += 1
        if c == 50:
            break
    model_speedups.append(speedups)
    cplex_gaps.append(cpx_gaps)
    mipgnn_gaps.append(gnn_gaps)
import numpy as np
print("mean speedup : ", np.mean(speedups))


import matplotlib.pyplot as plt
plt.figure()
plt.title("Speedup vs. default CPLEX")
plt.boxplot(model_speedups, labels=model_names, showfliers=False)
plt.ylabel("t_cplex / t")
plt.xticks(rotation = 45)
plt.tight_layout()

fig, axes = plt.subplots(2,1)
axes[0].set_title("CPLEX gap")
axes[0].boxplot(cplex_gaps, labels=model_names, showfliers=False)

axes[1].set_title("MIPGNN gap")
axes[1].boxplot(mipgnn_gaps, labels=model_names, showfliers=False)

plt.tight_layout()

plt.show()
exit()

num_jobs_final = 500
num_tasks = len(dict_list)
chunk_size = math.ceil(num_tasks / num_jobs_final)
chunk_size = 1
print("Chunks being mapped. chunk_size = %d" % chunk_size)
dict_listoflistsofdicts = list(chunks(dict_list, chunk_size))

timeout_min=math.ceil(math.ceil(timelimit/60.0)*chunk_size*1.2)
print("timeout_min = %d" % timeout_min)

#print(dict_listoflistsofdicts)
print(len(dict_listoflistsofdicts))
#exit()



print("Submitit initialization...")
executor = submitit.AutoExecutor(folder="slurm_logs_mipeval_new3")
print(executor.which())

executor.update_parameters(
        array_parallelism=1000,
        additional_parameters={"account": "rrg-khalile2"},
        timeout_min=timeout_min,
        mem_gb=mem_gb,
        cpus_per_task=num_cpus)

print("Submitit submitting job array...")
jobs = executor.map_array(combine_jobs, dict_listoflistsofdicts)
