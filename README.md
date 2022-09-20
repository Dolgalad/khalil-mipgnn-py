# MIP-GNN python implementation

Python implementation of the method presented in the paper "MIP-GNN: A Data-Driven Framework for Guiding Combinatorial Solvers" by
Khalil et al.

## Usage
This package allows you to generate MIP datasets, train MIP-GNN models and evaluate their performance on collections of MIP instances.
By default datasets are stored in `~/.cache/mipgnn/datasets` and models in `~/.cache/mipgnn/models`.

## Generating datasets
As of today this package only allows you to generate instance of Set Covering problems. 
```bash
python3 -m mipgnn.generate_data <dataset_name> -n <num_instances> -V <variables> -c <constraints> -d <density>
```

## Training a model
