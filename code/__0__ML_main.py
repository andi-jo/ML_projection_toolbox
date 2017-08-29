# -*- coding: utf-8 -*-
"""
MAIN: Case study UK CPI inflation projections
---------------------------------------------
from Bank of England SWP 674: Machine learning at central banks (September 2017)
- authors:         Chiranjit Chakraborty & Andreas Joseph
- disclaimer:      licence.txt and SWP 674 disclaimer apply
- documentation:   see README.txt for structure and comments for details
"""

#%% LOAD PACKAGES AND PROGRAM CONFIGURATION
# -----------------------------------------

from   aux__ML_import_packages      import *     # standard libraries
import aux__data_functions          as data_func # general data handling functions
import aux__ML_graphics             as ml_plot   # set of customised plots (heatmap, variable importance, etc.)
import aux__ML_functions            as ml_func   # machine learning wrapper and analysis tools

case = 'UK_CPI' # allows switching between different config-cases
#case = 'BJ_air'

# load config file
if   case=='UK_CPI':
    import config_UK_CPI            as config    # UK inflation projection
elif case=='BJ_air':
    import config_BJ_air            as config    # Beijing air pollution (for reference only)

#%% load and prepare data
#   ---------------------
import __A__ML_load_data            as data

#%% projections: training, cross-validation & testing in a projection setting
#   -------------------------------------------------------------------------
if config.do_projections==True:
    import __B__ML_projections      as pro # >>> time index hard-coding to match case
       
#%% analyse feature importance depending in lead-lag relation
#   ---------------------------------------------------------
if config.do_shift_anal==True:
    import __C__ML_shift_analysis   as sft

#%%  diagnostic plots for parts A-C
#    ------------------------------
if config.do_shift_anal==True:
    import __D__ML_diagnostics      as diag # >>> some case-dependent hard-coding for individual plots
    
