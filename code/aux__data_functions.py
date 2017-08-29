# -*- coding: utf-8 -*-
"""
Auxiliary data handling function: Case study UK CPI inflation projections
-------------------------------------------------------------------------
from Bank of England SWP 674: Machine learning at central banks (September 2017)
- authors:         Chiranjit Chakraborty & Andreas Joseph
- disclaimer:      licence.txt and SWP 674 disclaimer apply
- documentation:   see README.txt for structure and comments for details
"""

from __main__ import np,pd

def data_transformer(data,trafo,power=1.):
    """Transforms data including power.
     
    Parameters
    ----------
    data : 1-d numpy array
        data to be transformed
    
    trafo : str , format trafo-shift
        trafo legend
            NA  : none
            pow : power transformation
            log : logarithm base 10
            d1  : first difference
            pch : percentile change between subsequent elements
            ld  : log-difference
        sift only applies to 'd1', 'pch' and 'ld'
    
    power : float, optional (Default value = 1.)
        exponent (used as trafo+power)
    
    Returns
    -------
    1-d numpy array
        transformed data
    
    Raises
    ------
    ValueError
        Invalid transformation value.

    """
    
    tf = trafo.split('-')
    if tf[0]   =='NA': # only power transform
        return data
    elif tf[0] =='pow': # log_10 transform
        return data**power
    elif tf[0] =='log': # log_10 transform
        return np.log10(data)**power
    elif tf[0] =='d1': # first difference over period dx
        i=int(tf[1])
        return (data[i:]-data[:-i])**power
    elif tf[0] =='pch': # percentage change over period px
        i=int(tf[1])
        return (100.*(data[i:]-data[:-i])/data[:-i])**power
    elif tf[0] =='ld': # log difference (approx pch for small changes)
        i=int(tf[1])
        return (100*np.log(data[i:]/data[:-i]))**power
    else:
        raise ValueError("Invalid transformation value.")
        

def data_framer(data,target,features='all',index=None,start_i=None,end_i=None,shift=0,trafos=[],power=[],\
               name_trafo=True,drop_missing=True,write=False,out_name='output',CSV_input=True,\
               delimiter=',',CSV_output=True,in_sheet='Sheet1',out_sheet='Sheet1',\
               print_summary=False,corr_matrix=False,plot_data=False):
    """Select, transform and frame data.

    Parameters
    ----------
    data : pandas.DataFrame or filename
        input data form data frame or file
        
    target : str
        name of target variable (column name of data)
        
    features : list of str, optional (Default value = [])
        name of feature variables (if 'all', use data.columns, excl 'target')
        
    index : name, optional (Default value = None)
        name of index variable
        
    start_i : value, optional (Default value = None)
        index start of observations to be considered
        
    end_i : value, optional (Default value = None)
        index end of observations to be considered 
        
    shift : int, optional (Default value = 0)
        shift between target and features in units of index
        
    trafos : list of str, optional (Default value = [])
        transformations for each column in target and features
        
    power : list, optional (Default value = [])
        exponent of power transformations
        
    name_trafo : bool, optional (Default value = True)
        if True, include trafos in columns names of output frame
        
    drop_missing : bool, optional (Default value = True)
        if True, drop missing observations
        
    write : bool, optional (Default value = False)
        if True, write output frame to file
        
    out_name : str, optional (Default value = 'output')
        name of output file
        
    CSV_input : bool, csv input, optional (Default value = True)
        if True, csv-format expected, else Excel
        
    delimiter : str, optional (Default value = ',')
        columns separator
        
    CSV_output : bool, csv output, optional (Default value = True)
        if True, csv-format used, else Excel
        
    in_sheet : str, optional (Default value = 'Sheet1')
        name of input sheet for Excel format
        
    out_sheet : str, optional (Default value = 'Sheet1')
        name of output sheet for Excel format
        
    print_summary : bool, optional (Default value = False)
        if True, print summary statistics of output frame to screen
        
    corr_matrix : bool, optional (Default value = False)
        if True, print correlation matrix of output frame to screen
        
    plot_data : bool, optional (Default value = False)
        if True, plot output frame

    Returns
    -------
    pandas.DataFrame
        output data
        
    """
    
    # load dataframe from file (if not given)
    if type(data)==str: # load data if filename is given
        if CSV_input==True:
            data = pd.read_csv(data,sep=delimiter)
        else:
            data = pd.read_excel(data,in_sheet)
    
    # set index
    if not (data.index.name==index or index==None): # if not yet set
        data.set_index(index, inplace=True)
    
    # set start and end end indices
    if start_i==None:
        iS = 0
    else:
        try:
            iS = list(data.index).index(start_i)
        except ValueError:
            print 'Value or type of given start index value not matching index.'
            
    if end_i==None:
        iE = len(data.index)
    else:
        try:
            iE = list(data.index).index(end_i)
        except ValueError:
            print 'Value or type of given end index value not matching index.'
    
    # set feature variable
    if type(features)==str and not features=='all':
        features=[features]
    elif features=='all':
        features = list(data.columns)
        features.remove(target)
        features.remove(index)
    
    # set no trafos if empty lists are given
    if len(trafos)==0: # no level transformations
        trafos = ['NA' for i in range(len(features)+1)]
    if len(power)==0: # no power transformations
        power = np.ones(len(features)+1)
    
    # initiate new output dataframe
    data_new        = pd.DataFrame(columns=[index])
    data_new[index] = np.array(data.index[iS:iE+1])
    data_new.set_index(index, inplace=True)
    
    # get, slice and transform data (loop over target and features)
    for c,col in enumerate([target]+features):
        tf  = trafos[c].split('-')
        if len(tf)==1:
            t = 0
        else:
            t = int(tf[1])
        col_name = col
        if name_trafo==True:
            col_name += '-'+trafos[c]
        # target
        if c==0:
            if power[c]!=1 and name_trafo==True:
                col_name += '-E'+str(power[c])
            if ((iS-t)<0):
                raise ValueError('Target index transformation led to negative index.')
            data_slice         = np.array(data[col][iS-t:iE+1])
            data_new[col_name] = data_transformer(data_slice,trafos[c],power[c])
        # features
        else:
            if name_trafo==True:
                col_name += '-T'+str(shift)
            if power[c]!=1 and name_trafo==True: 
                col_name += '-E'+str(power[c])
            if ((iS-t-shift)<0):
                raise ValueError('Feature index shift or transformation led to negative index.')
            data_slice         = np.array(data[col][iS-t-shift:iE-shift+1])
            data_new[col_name] = data_transformer(data_slice,trafos[c],power[c])
    if drop_missing==True:
        data_new = data_new.dropna()
    # write new data to file
    if write==True:
        if CSV_output==True:
            data_new.to_csv(out_name,sep=delimiter)
        else:
            data_new.to_excel(out_name,out_sheet)
    # print summary stats of new data
    if print_summary==True:
        print '\nData summary:'
        print data_new.describe()
    # output correlation structure of new data
    if corr_matrix==True:
        print '\nData correlations matrix:'
        print data_new.corr()
    # plot new data
    if plot_data==True:
        df_plot = data_new.plot(lw=2)
        df_plot.legend(loc=2,prop={'size':9})
    return data_new


def get_alerts(df,features=None,cutoff_sides=None,n_min_alert=1,\
               p_cutoff=20,ID_name=None,add_alerts=True):
    """Generate outlier-based alerts for observations.
    
        Parameters
    ----------
    df : pandas.DataFrame
        input data form dataframe
        
    features : list, optional (Default value = None)
        list of names of columns in df to use for alerts
        
    cutoff_sides : str, optional (Default value = None)
        set outlier sides of distributions for features
        (L: left, R: right, LR: left & right)
        
    n_min_alert : int, optional (Default value = 1)
        minimal number of features outliers needed for overall alert of observation
        
    p_cutoff : float, optional (Default value = None)
        percentile cutoff to define outliers
        
    ID_name : str, optional (Default value = None)
        target or index colum neglected when creating alerts
        
    add_alerts : bool, optional (Default value = True)
        If True, alert column is added to df

    Returns
    -------
    dictionary
        incudes original data, alerts and alert stats
    
    """
    
    # get features and check names
    if features==None:
        features==df.columns
        if ID_name!=None:
            if ID_name in features:
                features = features[features!=ID_name]
    else:
        cols = df.columns
        for f in features:
            if not f in cols:
                raise ValueError('Got invalid feature name.')
    if (not ID_name==None) and (not ID_name in df.columns):
        raise ValueError('ID_name not in dataframe columns.')
    
    # cutoff sides
    if cutoff_sides==None:
        cutoff_sides = ['LR' for f in features]
    else:
        for f in range(len(features)):
            if not cutoff_sides[f] in ['LR','L','R']:# left-and-right, left-only, righ-only outliers
                raise ValueError('Got invalid value for cutoff side.')
                
    # get single outliers
    data       = df[features].copy()
    df_out     = df[features].copy()
    cut_values = np.zeros((len(features),2))*np.nan
    for i, (name,side) in enumerate(zip(features,cutoff_sides)):
        vals = data[name].values
        if    side=='LR':
            cut_values[i,0] = np.percentile(vals,p_cutoff/2,     interpolation='nearest') # LHS cutoff
            cut_values[i,1] = np.percentile(vals,100-p_cutoff/2, interpolation='nearest') # RHS
            df_out[name]  = (vals<=cut_values[i,0]) |  (vals>=cut_values[i,1]) # get cutoffs
        elif side =='L':
            cut_values[i,0] = np.percentile(vals,p_cutoff,       interpolation='nearest')
            df_out[name]  = vals<=cut_values[i,0]
        elif side =='R':
            cut_values[i,1] = np.percentile(vals,100-p_cutoff,   interpolation='nearest')
            df_out[name]  = vals>=cut_values[i,1]
    df_cutoffs = pd.DataFrame(cut_values,columns=['left','right'])
    df_cutoffs['features']  = features
    df_cutoffs.set_index('features',inplace=True)
    
    # get joint outliers
    M        = len(data)
    has_alert = np.zeros(M,dtype=bool)
    for r in range(M): # iterate over rows
        vals = df_out.iloc[r].values
        if np.sum(vals)>=n_min_alert:
            has_alert[r]=True
    alert_fraction = np.sum(has_alert,dtype=float)/len(has_alert)
    
    if add_alerts==True:
        alert_name = 'has-'+str(n_min_alert)+'-alerts'
        data[alert_name]  = has_alert
        df_out[alert_name] = has_alert
    
    if not ID_name==None:
        data[ID_name]  = df[ID_name]
        df_out[ID_name] = df[ID_name]
        
    out_dict = {'cutoffs'   : df_cutoffs,
                'has_alert'  : has_alert,
                'fraction'   : alert_fraction,
                'all_alerts' : df_out,
                'data'       : data}
               
    return out_dict


def is_iterable(thing):
    """Test of input is iterable.

    Parameters
    ----------
    thing : object
        input to be tested
        

    Returns
    -------
    bool, if True thing is iterable

    """
    
    try:
        iter(thing)
    except TypeError:
        return False
    return True


def to_zero_one(thing):
    """Set values to nearest zero or one value.

    Parameters
    ----------
    thing : number or array of numbers
        input data        

    Returns
    -------
    number or array of numbers with entries being either zero or one

    """
    
    if is_iterable(thing):
        zero_or_one = np.zeros(len(thing))
        zero_or_one[np.array(thing)>=0.5]=1
    else:
        if thing>=0.5:
            zero_or_one = 1
        else:
            zero_or_one = 0
    return zero_or_one



def compare_LR(value,val_L=0,val_R=0,side='LR'):
    """Check if value is beyond left/right boundary values.
    
    Parameters
    ----------
    value : float
        value to compare to val_L, val_R
        
    val_L : float
        left comparison value
        
    val_R : float
        right comparison value
        
    side : str (L,R or LR)
        indicate side of comparison: left (L), right (R) or both (LR)

    Returns
    -------
    bool
    
    
    """
    
    if len(side)==2:
        is_beyond = (value<val_L) | (value>val_R)
    elif side=='L':
        is_beyond = (value<val_L)
    elif side=='R':
        is_beyond = (value>val_R)
    else:
        raise ValueError('Invalid side given.')
        
    return is_beyond
    
    
