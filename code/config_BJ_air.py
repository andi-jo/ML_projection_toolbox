# -*- coding: utf-8 -*-
"""
CONFIGURATIONS: Beijing air pollution level prediction
------------------------------------------------------

Alternatively provided configuration file to better understand program structure and options. 

- disclaimer:      licence.txt. Case study has not been calibrated in any way to produce sensible results 
                   but to demonstrate the working and options of the overall wrapper only. 
- documentation:   see README.txt for structure and comments for details

NOTE: There is no guarantee that every combination of settings will execute without causing an error. 
      The default settings in this config-file have been tested and execute correctly, such that an original copy may be kept.

"""


# paths
main_path         = 'X:\\your\\path\\'
data_path         = main_path+'data\\'    # data dir
code_path         = main_path+'code\\'    # code dir
out_path          = main_path+'results\\' # output dir
fig_path          = main_path+'figures\\' # figure output dir

# DATA
# ----
# input
application       = 'Forecasting air pollution in Beijing (Jan 2010 - Dec 2014)' # short program description
datafile          = 'Beijing_PM2.5_wLvl' # name of data file
file_format       = 'csv'      # 'csv' (comma separator) or Excel ('xls/x', first sheet)
description       = 'hourly data (Jan 2010 - Dec 2014)'

# time range
time_var          = 'rangeL'    # name of time index variable (must be column in 'datafile') 
                              # or 'rangeL': use range(len(data)) as index
unit              = 'hours'  # max=43824
start_time        = 30000       # data start value of 'time_var' (should be int, float or str type)
end_time          = 40000   # data end   value of 'time_var' (should be int, float or str type)


# Variables (LHS, RHS, projection horizon, transformations, normalisation)
target            = 'pm2.5_above100'  # LHS variable
horizon           = 24      # time horizon of projections in units of 'time_var' (here quarter; LHS-RHS shift in lead-lag model)


# list of features (RHS) to be considered (must be columns in 'datafile')
features          = 'all' # all: ['year','month','day','hour','DEWP','TEMP','PRES','cbwd','Iws','Is','Is']
categorical       = ['cbwd']  # feature for one-hot encoding of categorical variable
to_norm           = ['TEMP','Iws'] # features to norm (z-scores; if not None, requires list of feature names to be normed)
                            # may induce look-ahead bias in time series setup (full dataset used for normalisation)
# transformations for all columns in "datafile" (e.g. first difference, percentage change, etc.)
data_trafos       = None # None, list ordered as data.columns or dictionary keyed by feature names



# DATA (type, test fraction, number of bootstraps)
# -------------------------------------------------
method            = 'Tree-clf' # 'NB' (Naive-Bayes),'SVM','NN' (neural net), 'Tree','Forest','kNN','Logit','OLS','VAR'
                            # suffix: 'rgr' : regression problem, 'clf' : classification problem
is_class          = True # if model should treat output as class (class labels need to be given as integer values)
ref_model         = None # only VAR supported or None.
                            #         NB only takes clf, No suffix for Logit, OLS and VAR
test_fraction     = 0.3         # fraction of dataset used for testing
n_boot            = 200        # number of bootstraps (sampled models)


# PRECEDURES & OPTIONS
# --------------------

# general
do_model_fit      = True        # modeling training/testing (if False, tries to load pre-computed results)
counter_fact      = False        # if True, evaluate variable importance by leaving each variable out and compare to full model
                            # if False, only recorded for tree models

# projections
do_projections    = True        # modelling setting for projection
time_step_size    = 5000           # step in units if 'time_var' by which window expands or moved
init_train_period = time_step_size # set 0 for full horizon, cross-section)
fixed_start       = False        # if True: expanding horizon, if False: sliding window
full_model_proj   = True        # use full data set for training for final projection
ptile_cutoff      = 10.         # total p-tile range (%) to be cut off (1/2 on each side), error band if bootstrapped model

# variable contribitions for different lead-lag relations (horizons)
do_shift_anal     = True       # investigate horizon dependence (LHS-RHS shift) of variable dependence
time_shifts       = range(0,121,12) # list of shifts to investigate in horizon analysis (units of 'time_var')

# cross-validation (CV) setting
# NOTE: standard settings after CV need to be added to 'aux__ML_functions.py' and analysis rerun.  
#       modify function "modelSelection()" to customise CV
CV_name         = 'max_depth'       # if not None, insert name of hyper-parameter (str) for CV across CVvalues 
CV_values       = range(6,19,2) # cross-validation values (no zero allowed)
CV_at_last      = True       # if False, only use initial training period for CV. 
                            #If True, final data set (potential look-ahead bias)  

# output options
verbose         = False       # print function log to screen (e.g. intermediate fit results)
do_plot_diag    = True        # plot diagnostic graphs
save_plots      = True
save_results    = False        # whether to save results and plots or not
save_models     = True        # if False, no fitted model objects are saved (may save lots of memory, 
                              # but also lead to errors if models are needed as input)
name_add        = '_test_1'      # custom identifier
fig_format      = 'png'        # format figures are saved in


