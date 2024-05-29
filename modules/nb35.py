#!/usr/bin/env python
# coding: utf-8

# ### python scripts

# - scripts are code files to be executed
# - we can use scripts also to define our own ***python modules*** with functions or classes that we can afterwards import in other scripts or notebooks

# In[1]:


import numpy as np
import pandas as pd


# #### define a function that generates a dataset to use in our notebooks

# In[2]:


def dataset_1(N = 100, m = (10, 5, 3)):
    # draw N samples of 3 continuous variables from a Dirichlet distribution
    data_C = pd.DataFrame(np.random.default_rng(seed = 1234).dirichlet(m, N), columns = ['C%d' %j for j in range(len(m))])
    # draw N samples of m discrete variables from a Dirichlet distribution
    data_D = pd.DataFrame((np.random.default_rng(seed = 8672).dirichlet(m, N) *10).astype('int'), columns = ['D%d' %j for j in range(len(m))])
    # map categorical values
    cats = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for j in range(len(m)):
        data_D['D%d' %j] = data_D['D%d' %j].map({x:cats[i] for i, x in enumerate(np.unique(data_D['D%d' %j]))})
    return pd.concat((data_C, data_D), axis = 1)


# df = dataset_1()
# df.head()

# #### Define a 2D contingency table class

# In[4]:


class ContingencyTable2D():
    
    def __init__(self, X, Y):
 
        # cardinalities
        self.cardX = np.unique(X).shape[0]
        self.cardY = np.unique(Y).shape[0]
    
        # factorize
        X_ = X.map({x: i for i,x in enumerate(np.unique(X.sort_values()))})
        Y_ = Y.map({y: j for j,y in enumerate(np.unique(Y.sort_values()))})
        
        # joint counts
        self.counts = np.zeros((self.cardX, self.cardY))
        for x, y in zip(X_, Y_): self.counts[x, y] += 1

        #total counts
        self.n = np.sum(self.counts)
        
    def mrgX(self):
        return np.sum(self.counts, axis = 1)

    def mrgY(self):
        return np.sum(self.counts, axis = 0)


# ct = ContingencyTable2D(df.D0, df.D1)
# ct.counts

# ct.mrgX()

# ct.mrgY()

# ct.n
