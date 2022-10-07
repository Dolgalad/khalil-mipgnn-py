import os
import argparse
import json

from mipgnn.datasets import list_processed_datasets
from mipgnn.gnn_models import manager as gnn_manager

parser = argparse.ArgumentParser()
parser.add_argument("--list-datasets", action="store_true", help="show a list of processed datasets")
parser.add_argument("--list-gnn-models", action="store_true", help="show a list of available trainable GNNs")
parser.add_argument("--list-models", action="store_true", help="show a list of available trained models")
parser.add_argument("--model-info", type=str, help="show information about a model")



args = parser.parse_args()

def trained_model_list():
    models_directory = os.path.expanduser("~/.cache/mipgnn/models")
    models = list([f for f in os.listdir(models_directory) if not f.endswith(".log")])
    return models

def get_model_info(model_name):
    models_directory = os.path.expanduser("~/.cache/mipgnn/models")
    model_info_path = os.path.join(models_directory, model_name+".json")
    if not os.path.exists(model_info_path):
        return {}
    with open(model_info_path, "r") as f:
        return json.loads(f.read())



if args.list_datasets:
    print("Processed datasets : ")
    for dataset in list_processed_datasets():
        print(f"\t{dataset}")

if args.list_gnn_models:
    print("#GNN Models:")
    print("#\tprefix\tname")
    for i,cls in enumerate(gnn_manager.list_model_classes()):
        print(f"{i+1}\t{cls._class_prefix}\t{cls.__name__}")

if args.list_models:
    print("Trained models")
    models = trained_model_list()#list([f for f in os.listdir(models_directory) if not f.endswith(".log")])
    for model in models:
        print(f"\t{model}")

if args.model_info:
    models = trained_model_list()
    if not args.model_info in models:
        print(f"Model {args.model_info} does not exists")
    model_data = get_model_info(args.model_info)
    if model_data:
        print(json.dumps(model_data, indent=4, sort_keys=True))
    
    
