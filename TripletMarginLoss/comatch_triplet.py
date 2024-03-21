# File: comatch_triplet.py
# Authored by: https://github.com/rufernan

"""
## CoMatch Trainer Modification (Classification-SemiCLS/trainer/comatch.py)

### Objective
The modifications introduced in this file enhance CoMatch by adding support for deep metric learning.

### Dependencies
1. This implementation is based on the [TencentYoutuResearch/Classification-SemiCLS](https://github.com/TencentYoutuResearch/Classification-SemiCLS) repository.
2. Utilizes the [PyTorch Metric Learning](https://kevinmusgrave.github.io/pytorch-metric-learning/) library.

"""
# Import libraries
from spectral import *
from pytorch_metric_learning import distances, reducers, losses, miners

# Add initialization for deep metric learning
self.lambda_dml = self.cfg.lambda_dml  # for deep metric learning

# Introduce the following code in replacement of the following line
# 169: loss = loss_x + self.lambda_u * loss_u + self.lambda_c * loss_contrast

"""
#### Triplet Margin Loss for Deep Metric Learning
"""
distance = distances.CosineSimilarity()  # Define cosine similarity distance 
reducer = reducers.ThresholdReducer(low=0)  # Define threshold reducer
loss_func = losses.TripletMarginLoss(margin=0.2, distance=distance, reducer=reducer)  # Initialize triplet margin loss
mining_func = miners.TripletMarginMiner(margin=0.2, distance=distance, type_of_triplets="semihard")  # Initialize triplet margin miner

# Prepare embeddings and labels
embeddings = feats_u_s0  # Base embedding for deep metric learning (used for loss_u)
labels = lbs_u_guess  # Pseudo-labels for batch used in deep metric learning loss ()

# Mining triplets
indices_tuple = mining_func(embeddings, labels)

# Compute deep metric learning loss
loss_dml = loss_func(embeddings, labels, indices_tuple)

# Integrate deep metric learning loss into the overall loss function
loss = loss_x + self.lambda_u * loss_u + self.lambda_c * loss_contrast + (self.lambda_dml * loss_dml)
