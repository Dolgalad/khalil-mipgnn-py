# Installation of the environment

`mipgnn` depends on the following : 
  - CPLEX (with python package)
  - SCIP optimization suite
  - `ecole` python package
  

## CPLEX 
Visit the [CPLEX homepage](https://www.ibm.com/analytics/cplex-optimizer) for instructions on installing CPLEX Optimizer.

## SCIP
Visit the [SCIP homepage](https://scipopt.org) for instructions on installing SCIP Optimization Suite.

Edit the `SCIP_DIR` environment variable in `ecole_env.sh` file: 
```
export SCIP_DIR=<path/to/SCIPOptSuite-8.0.1-Linux>
```

## Ecole
See the `ecole_installation.txt` file for instructions on how to install `ecole`.

## Requirements
Create a virtual environment and activate :
```bash
python3 -m venv venv
. venv/bin/activate
python -m pip install -r requirements.txt
```


