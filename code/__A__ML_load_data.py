# -*- coding: utf-8 -*-
"""
PART A - DATA IMPORT: Case study UK CPI inflation projections
-------------------------------------------------------------
from SWP XXX, Machine learning at central banks, Bank of England, 2017.
Authors: Chiranjit Chakraborty & Andreas Joseph.

See README.txt for details
"""
from __main__ import config,data_func,pd,np,time,pat

time_stamp = time.strftime('%Y-%m-%d %H:%M:%S')

# print settings to screen
print '\n\n'
print '    =================================='
print '    ||   MACHINE LEARNING WRAPPER   ||'
print '    ||   for projection analyses    ||' 
print '    ==================================\n'

print 'Application: {0}\n'.format(config.application)
print 'time: {0}\n'.format(time_stamp)
print '\nProgram settings\n----------------\n'
print '\ttarget             : {0}'.format(config.target)
print '\tmethod             : {0}'.format(config.method)
print '\thorizon            : {0} ({1})'.format(config.horizon,config.unit)
print '\tbootstraps         : {0}'.format(config.n_boot)
print '\ttest fraction      : {0}'.format(config.test_fraction)
print '\tdata period        : {0} - {1}'.format(config.start_time,config.end_time)
print '\tinit. train period : {0} ({1})'.format(config.init_train_period,config.unit)
print '\ttime step size     : {0} ({1})'.format(config.time_step_size,config.unit)
print '\tfixed window       : {0}'.format(config.fixed_start)
print '\tdata description   : {0}\n'.format(config.description)


#%% load data and transformations
datafile = config.data_path+config.datafile+'.'+config.file_format

if config.file_format=='csv':
    raw_data = pd.read_csv(datafile)
elif config.file_format=='xls' or config.file_format=='xlsx':
    raw_data = pd.read_excel(datafile)
else:
    raise ValueError('\nNo valid file format given.\n')

# default index
if config.time_var=='rangeL':
    raw_data[config.time_var] = range(len(raw_data))

# read all features from data 
if config.features=='all': 
    config.features = list(raw_data.columns)
    config.features.remove(config.target)
    config.features.remove(config.time_var)

# data column transformations
if config.data_trafos==None:
    trafos = ['NA' for name in [config.target]+config.features]
elif type(config.data_trafos)==dict:
    trafos = [config.data_trafos[name] for name in [config.target]+config.features]
elif data_func.is_iterable(config.data_trafos)==True:
    trafos = [config.data_trafos[i] for i in range(len([config.target]+config.features))]
else:
    raise ValueError('Invalid data transformation type.')
    

# 1-hot-encoding of categorical variables
for cat in config.categorical:
    # data trafos
    i_cat_trafo = int(np.where(np.array(config.features)==cat)[0])
    cat_trafo   = trafos[i_cat_trafo]
    del trafos[i_cat_trafo]
    # get indicator frames
    cat_rawData  = pat.dmatrix(cat, raw_data, return_type='dataframe').iloc[:,1:]
    # append columns
    for col in cat_rawData.columns:
        raw_data[col] = cat_rawData[col]
        config.features.append(col)
        trafos.append(cat_trafo)
    config.features.remove(cat)


#%% get transformed data
data_shifted = data_func.data_framer(data=raw_data.copy(),target=config.target,features=config.features,\
                                     index=config.time_var,start_i=config.start_time,end_i=config.end_time,\
                                     shift=config.horizon,trafos=trafos,name_trafo=False,drop_missing=True)

# number of observations trasining data length
M = len(data_shifted)
if M<10: # thin dataset warning
    print 'Warning: Dataset has less than 20 observations.\n'

if config.init_train_period==0: # use full data set from the start
    config.init_train_period = M

# test data, where features have not been shifted relative to target (needed for future projections).
data_no_shift = data_func.data_framer(data=raw_data.copy(),target=config.target,features=config.features,\
                                      index=config.time_var,start_i=config.start_time,end_i=config.end_time,\
                                      shift=0,trafos=trafos,name_trafo=False,drop_missing=True)

# identifiers for later use
ID_short  = '{0}_{1}_{2}-{3}-shift'.format(config.target,config.method,config.horizon,config.unit)
ID_long   = ID_short+'_fixedStart-{0}_trainStartPeriod-{1}_{2}'.format(config.fixed_start,config.init_train_period,config.unit)
ID_long  += '_crossVal-{0}_fullDataCV-{1}_{2}'.format(config.CV_name,config.CV_at_last,config.name_add)
ID_short += '_date-{0}_{1}'.format(time_stamp[:10],config.name_add)

print '\nData loaded successfully.\n'