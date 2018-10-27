"""
Module for some useful functions
"""

import numpy as np

def soft_max(scores, y):
    """
    compute the sofmax loss function and it's gradient in 
    respect the score
    """

    N,D = scores.shape
    #shifted scores
    sh_scores = scores - np.max(scores,axis=1,keepdims=1)

    #taking the exponential
    sh_scores = np.exp(scores)

    #normalizing
    normalized_scores = sh_scores / sh_scores.sum(axis=1,keepdims=True)

    #computing the gradient
    grad = scores -(1 - np.sum(scores,axis=1,keepdims=1))

    
    
    #taking the score over y
    I = np.arange(scores.shape[0])
    normalized_scores = -np.log( normalized_scores[I,y])

    #computing the loss
    loss = np.mean(normalized_scores)



    return loss,grad


