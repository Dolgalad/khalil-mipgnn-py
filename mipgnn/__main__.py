import os
import argparse

from mipgnn.datasets import list_processed_datasets
from mipgnn.gnn_models import list_gnn_models

parser = argparse.ArgumentParser()
parser.add_argument("--list-datasets", action="store_true", help="show a list of processed datasets")
parser.add_argument("--list-gnn-models", action="store_true", help="show a list of available trainable GNNs")


args = parser.parse_args()

if args.list_datasets:
    print("Processed datasets : ")
    for dataset in list_processed_datasets():
        print(f"\t{dataset}")

if args.list_gnn_models:
    print("GNN Models")
    for name,_ in list_gnn_models():
        print(f"\t{name}")
    
