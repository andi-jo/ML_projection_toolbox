# -*- coding: utf-8 -*-
"""
CONFIGURATIONS: Case study UK CPI inflation projections
-------------------------------------------------------
from SWP XXX, Machine learning at central banks, Bank of England, 2017.
Authors: Chiranjit Chakraborty & Andreas Joseph.

See README.txt for details

NOTE: There is no gurantee that every combination of settings will execute without causing an error. 
      The default settings in this config-file have been tested and execute correctly, such that an original copy may be kept.

"""

# paths
#main_path        = 'X:\\_People\\Andreas Joseph\\research projects\\machine learning at central banks\\'
#main_path        = 'C:\\Users\\Andreas_Joseph\\Dropbox\\boe\\projects resources\\ML paper\\code\\working folder - public\\'
main_path        = 'C:\\Users\\Andreas Joseph\\Dropbox\\boe\\projects resources\\ML paper\\code\\working folder - public\\'
data_path        = main_path+'data\\'    # data dir
code_path        = main_path+'code\\'    # code dir
out_path         = main_path+'results\\' # output dir
fig_path         = main_path+'figures\\' # figure output dir

# DATA
# ----
# input
application      = 'UK CPI inflation modelling' # short program description
datafile         = 'UK_DATA_23Jan17' # name of data file
file_format      = 'xlsx'      # 'csv' (comma separator) or Excel ('xls/x', first sheet)
description      = 'quarterly macro time series'

# time range
time_var         = 'date'     # name of time index variable (must be column in 'datafile')
                              # or 'rangeL': use range(len(data)) as index
unit             = 'quarters'
start_time       = '1988Q1'   # data start value of 'time_var' (should be int, float or str type)
end_time         = '2015Q4'   # data end   value of 'time_var' (should be int, float or str type)
break_point      = '2008Q3'  # onset of global financel crises (custom variable for this case)

# Variables (LHS, RHS, projection horizon, transformations, normalisation, time/index variable)
target           = 'CPI'       # LHS variable
horizon          = 8          # time horizon of projections in units of 'time_var' (here quarter; LHS-RHS shift in lead-lag model)

# list of features (RHS) to be considered (must be columns in 'datafile')
features         = ['M4','Private_debt','Employment_rate','Unemployment','GDP','GDHI',\
                   'Labour_prod','Bank_rate','5YrBoYldImplInfl','ERI','Comm_idx']
categorical      = []          # feature for one-hot encoding of categorical variable
to_norm          = None        # features to norm (z-scores; if not None, requires list of feature names to be normed)
                            # may induce look-ahead bias in time series setup (full dataset used for normalisation)

# data column transformations: None, list ordered as data.columns or dictionary keyed by target+feature names
# needs to be compatible with function: data_func.dataTransformer, e.g. 'NA', 'log' or 'pch-4'.
h = str(int(horizon)) # horizon parameter as string (for convenience)
data_trafos      = ['pch-'+h,'pch-'+h,'pch-'+h,'pch-'+h,'NA','pch-'+h,'pch-'+h,\
                    'pch-'+h,'NA','NA','pch-'+h,'pch-'+h]


# DATA (type, test fraction, number of bootstraps)
# -------------------------------------------------
method           = 'Tree-rgr' # 'NB' (Naive-Bayes),'SVM','NN' (neural net), 'Tree','Forest','kNN','Logit','OLS','VAR'
                            # suffix: 'rgr' : regression problem, 'clf' : classification problem
is_class         = False # if model should treat output as class (class labels need to be given as integer values)
ref_model        = 'VAR'#'VAR' # only VAR supported or None.
                            #         NB only takes clf, No suffix for Logit, OLS and VAR
test_fraction    = 0.3         # fraction of dataset used for testing
n_boot           = 1000        # number of bootstraps (sampled models)


# PRECEDURES & OPTIONS
# --------------------

# general
do_model_fit     = True        # modeling training/testing (if False, tries to load pre-computed results)
counter_fact     = False        # if True, evaluate variable importance by leaving each variable out and compare to full model
                               # if False, only recorded for tree models

# projections
do_projections    = True        # modelling setting for projections
time_step_size    = 4           # step in units if 'time_var' by which window expands or moved
init_train_period = 40          # number of time steps to start expanding horizon with (set 0 for full horizon, cross-section)
fixed_start       = False        # if True: expanding horizon, if False: sliding window
full_model_proj   = True        # use full data set for training for final projection
ptile_cutoff      = 10.         # total p-tile range (%) to be cut off (1/2 on each side), error band if bootstrapped model

# variable contributions for different lead-lag relations (horizons)
do_shift_anal     = True       # investigate horizon dependence (LHS-RHS shift) of variable dependence
time_shifts       = range(0,19,2) # list of shifts to investigate in horizon analysis (units of 'time_var')

# cross-validation (CV) setting
# NOTE: standard settings after CV need to be added to 'aux__ML_functions.py' and analysis rerun.  
#       modify function "modelSelection()" to customise CV
CV_name           = None       # if not None, insert name of hyper-parameter (str) for CV across CV values 
CV_values         = None        # cross-validation values (no zero allowed)
CV_at_last        = False       # if False, only use initial training period for CV. 
                                #If True, final data set (potential look-ahead bias)  

# output options
verbose         = False       # print function log to screen (e.g. intermediate fit results)
do_plot_diag    = True        # plot diagnostic graphs
save_plots      = True
save_results    = False        # whether to save results and plots or not
save_models     = True        # if False, no fitted model objects are saved (may save lots of memory, 
                              # but also lead to errors if models are needed as input)
name_add        = '_test_1'      # custom identifier
fig_format      = 'png'       # format figures are saved in
