# -*- coding: utf-8 -*-
"""
Auxiliary package import: Case study UK CPI inflation projections
-----------------------------------------------------------------
from Bank of England SWP 674: Machine learning at central banks (September 2017)
- authors:         Chiranjit Chakraborty & Andreas Joseph
- disclaimer:      licence.txt and SWP 674 disclaimer apply
- documentation:   see README.txt for structure and comments for details
"""


# basics
import numpy              as np
import pandas             as pd
import matplotlib.pyplot  as plt
import matplotlib.colors  as colors
import scipy.stats        as st
import matplotlib.cm      as cmx
import random             as rn
import pickle             as pk
import matplotlib.patches as patch
import random             as rdm
import patsy              as pat



# machine learning (from scikit-learn)
import sklearn.base             as skl_base
import sklearn.ensemble         as skl_ens
import sklearn.neural_network   as skl_nn
import sklearn.tree             as skl_tree
import sklearn.linear_model     as skl_lin
import sklearn.neighbors        as skl_neigh
import sklearn.svm              as skl_svm
import sklearn.naive_bayes      as skl_NB
import sklearn.metrics          as skl_metrics

# some extras
import statsmodels.tsa.vector_ar.var_model as sm_vars
import re
import time
import os
import warnings
warnings.filterwarnings('ignore')
