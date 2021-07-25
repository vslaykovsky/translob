# Intro

This is an [unsuccessful] attempt to reproduce results from the paper _Transformers for Limit Order Books_
See python/train.py and python/translob.ipynb for implementation of the model. 
Model trained with the best hyperparameters from the paper only produces 48 average F1 score as opposed to 91.61 promised in the paper.  


# Code repository for TransLOB 

This is the repository for the paper _Transformers for Limit Order Books_ which uses a CNN for feature extraction followed by a Transformer to predict future price movements from limit order book data. 

Paper :  

* ``TransLOB.pdf``.

Python files :  

* ``LobFeatures.py`` causal convolutional function.
* ``LobAttention.py`` multi-head self-attention function.
* ``LobPosition.py`` positional encodings.
* ``LobTransformer.py`` transformer function.

This is research code and some assembly may be required.


# FI-2010 dataset

The FI-2010 dataset is made up of 10 days of 5 stocks from the Helsinki Stock Exchange, operated by Nasdaq Nordic, consisting of 10 orders on each side of the LOB. Event types can be executions, order submissions, and order cancellations and are non-uniform in time. We restrict to normal trading hours (no auction).

There are 149 rows in each file : 40 LOB data points, 104 hand-crafted features and 5 prediction horizons (k=10, 20, 30, 50, 100).
Each column represents an event snapshot. Data is normalized based on the prior day mean and standard deviation and is stored consecutively for each of the 5 stocks. 

The training labels for prediction are as follows. Let a = 0.002. For percentage changes x >= 0.002, label 1.0 is used. For percentage changes -a < x < a, label 2.0 is used. For percentage changes x <= -a, label 3.0 is used.


# Attention visualization

Convolutional output with position encoding.  

<p align="center">
<img src="https://github.com/jwallbridge/translob/blob/master/figures/conv1.png" width="75%" height="75%">
</p>

Attention distributions in head 1 in the first transformer block.  

<p align="center">
<img src="https://github.com/jwallbridge/translob/blob/master/figures/block1head1.png" width="50%" height="50%">
</p>

Attention distributions in head 2 in the first transformer block.  

<p align="center">
<img src="https://github.com/jwallbridge/translob/blob/master/figures/block1head2.png" width="50%" height="50%">
</p>

Attention distributions in head 3 of the first transformer block.  

<p align="center">
<img src="https://github.com/jwallbridge/translob/blob/master/figures/block1head3.png" width="50%" height="50%">
</p>
