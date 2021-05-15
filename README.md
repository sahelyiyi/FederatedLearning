# Federated Multi-Task Learning from Big Data over Networks

This repository is the official implementation of [Federated Multi-Task Learning from Big Data over Networks](https://arxiv.org/abs/2030.12345) paper. 

![alt text](algorithm1.png)

The package related to the algorithm implementation is `algorithm` which has the following structure.

* main.py 
    * the implementation of algorithm 1
* optimizer.py 
    * the primal optimizer for the loss functions, we have implemented 
    the squared error loss, and the logistic loss with `pytorch` 
    and also the straight forward primal optimizer in the 
    Networked Linear Regression section which is much faster than the `pytorch` one
* penalty.py
    * the different penalty functions for the dual optimizers explained in the paper, 
    `norm2`, `MOCHA`, and `norm1`
 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

>📋  For running the code you need to have python 3.7

## Experiments

In this repository we have multiple experiments

### Stochastic Block Model



Create SBM
Creating the SBM with two clusters 
<img src="https://render.githubusercontent.com/render/math?math=|C_1| = |C_2| = 100"> .

Two nodes within the same cluster are connected by an edge with probability `pin`, 
Two nodes from different clusters are connected by an edge with probability `pout`. 
Each node <img src="https://render.githubusercontent.com/render/math?math=i \in V"> represents a local dataset consisting of 
`m` feature vectors 
<img src="https://render.githubusercontent.com/render/math?math=x^{(i,1)} , . . . , x^{(i,m)} \in R^n"> . 
The feature vectors are i.i.d. realizations of a standard Gaussian random vector 
<img src="https://render.githubusercontent.com/render/math?math=x \sim N(0,I)">. The labels 
<img src="https://render.githubusercontent.com/render/math?math=y^{(i)}_1,...,y^{(i)}_m \in R">
 of the nodes Each node <img src="https://render.githubusercontent.com/render/math?math=i \in V">
are generated according to the linear model 
<img src="https://render.githubusercontent.com/render/math?math=y^{(i)}_r = (x^{(i,r)})^T \bar{w}^{(i)} + \epsilon"> , 
with <img src="https://render.githubusercontent.com/render/math?math=\epsilon \sim N(0,\sigma)">, 
weight vector 
<img src="https://render.githubusercontent.com/render/math?math=w^{(i)} = (2, 2)^T">
for <img src="https://render.githubusercontent.com/render/math?math=i \in C_1"> ,  and 
<img src="https://render.githubusercontent.com/render/math?math=w^{(i)} = (-2, 2)^T">
for for <img src="https://render.githubusercontent.com/render/math?math=i \in C_2">.

To train the model(s) in the paper, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
