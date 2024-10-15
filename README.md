# GT-NAC Experiments Code

## Description

This repository contains the source code for the experiments done for the paper "An Optimized Graph Transformer Network Attack Classifier".

## Steps to run the code:

1. Go to directory ../Datasets where you should follow the instructions to download and properly place the files from the datasets you want.
2. The code for graph_approaches and flow_approaches uses separate conda environments. Using conda and the files graph_approaches/torch_environment.yml and flow_approaches/tf_environment you can import the conda environments used to run the code. For more info on conda, look up anaconda or miniconda. The command to import a conda environment is "conda env create -f <environment file>".
3. Activate the conda environment you imported with the "conda activate" command. Then run the command "jupyter notebook".
4. Select the notebook you want and run the appropriate notebook.

## Extra Information

1. The graph models create checkpoints into the ../Checkpoints directory, and results in Pickle files and Confusion Diagrams into the Results directory. The flow models show all results in the notebook.
2. The graph models in the first cells of the notebooks preprocess the PCAPs into CSVs and into graphs. These steps store the results in files, and can be omitted in following executions.
3. If you get "ERROR: ModuleNotFoundError: No module named 'torch_geometric.utils.to_dense_adj'" when trying to run the temporal models, you need to go to file: ~/miniconda3/envs/torch/lib/python3.12/site-packages/torch_geometric_temporal/nn/attention/tsagcn.py and change the line "from torch_geometric.utils.to_dense_adj import to_dense_adj" to "from torch_geometric.utils import to_dense_adj". This will fix the issue.
