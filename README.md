# Federated Learning

This project contains the implementations of [FEDERATED LEARNING FROM BIG DATA OVER NETWORKS](https://arxiv.org/pdf/2010.14159.pdf) paper.


## Folders and Files Guideline
To empirically evaluate the accuracy of Algorithm 1, we two different types of 
experiments.The first one is related to linear regression applications and 
the other one is related to the transfer learning applications.

1. deep_learning_lasso/ : Implementations related to the transfer learning experiments.
     - deep_learning_utils.py and deep_learning_utils.py : utils functions related to the experiment.
     - main.py : load the saved data fro training the models, run algorithm1 on it and 
     then save the figures related to it.
     - models.py : Implementations for the base and the new model.
     - scores.py : save the figures related to the experiment (the orange and blue figure).
     -  train.py : Train the model for different training datasets and save the results.
     
2. regression_lasso/ : Implementations related to the regression experiments.
     - main.py : get the graph data as the input, run algorithm1 on it and 
     then return the scores for it.
     - reg_complete.py : Create a complete graph and then run algorithm1 for it.
     - reg_merge_3d_road.py : Create the graph related to 
     [3D-road dataset](https://archive.ics.uci.edu/ml/datasets/3D+Road+Network+%5C%28North+Jutland%5C%2C+Denmark%5C%29)
     and then run algorithm1 for it.
     - reg_sbm.py : Create the graph for SBM two clusters and also five clusters and 
     then run algorithm1 for each of them.
     - scores.py : Calculate the scores for the experiment after running algorithm1 for it.
     
3. algorithm.py : Implementation of algorithm1.
4. experiments.ipynb : the notebook related to the experiments.
5. stochastic_block_model.py : Create the graph related to the SBM.

