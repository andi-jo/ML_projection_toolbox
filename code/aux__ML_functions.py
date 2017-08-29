# -*- coding: utf-8 -*-
"""
Auxiliary functions: Case study UK CPI inflation projections
------------------------------------------------------------
from Bank of England SWP 674: Machine learning at central banks (September 2017)
- authors:         Chiranjit Chakraborty & Andreas Joseph
- disclaimer:      licence.txt and SWP 674 disclaimer apply
- documentation:   see README.txt for structure and comments for details
"""

from __main__ import np,st,rn,pk,rdm,data_func,\
                     skl_base,skl_ens,skl_nn,skl_tree,skl_lin,\
                     skl_neigh,skl_svm,skl_NB,skl_metrics,sm_vars\

def ML_train_tester(df,target,features,method,m_test=1,n_boot=500,is_class=False,is_zero_one=False,\
                   to_norm=None,CV_name=None,CV_value=None,counter_fact=False,\
                   horizon=None,save_out=False,save_models=True,file_name='',verbose=False):
    """Machine learning wrapper for bootstrapped training and testing.

    Parameters
    ----------
    df : pandas.DataFrame (input data)
        
    target : str
        LHS variable
        
    features : list of str
        RHS variable(s)
        
    method : str
        model
        
    m_test : int or index mask, optional (Default value = 1, "jackknife")
        size of test data or index max of test set. If mask, n_boot is set to 1
        
    n_boot : int, optional (Default value = 500)
        number of bootstraps
        
    is_class : bool, optional (Default value = False)
        if True, maps to integer output
        
    is_zero_one : bool, optional (Default value = True)
        if True, maps to Boolean output
        
    to_norm : list, optional (Default value = None)
        variables to norm (z-scores)
        
    CV_name : str, optional (Default value = None)
        name of cross-validation parameter
        
    CV_value : float, optional (Default value = None)
        value for cross-validation parameter
        
    counter_fact : bool, optional (Default value = False)
        if True, variable importance by leaving one feature out at a time
        
    horizon : int, optional (Default value = None)
        lead-lag size for projection model (only used for VAR)
        
    save_out : bool, optional (Default value = False)
        if True save output to file
        
    save_models : bool, optional (Default value = True)
        if True, include  models in output file (could use lots of space)

    file_name : str, optional (Default value = '')
        name of output file
    verbose : bool, optional (Default value = False)
        if True, print basic fit results to screen

    Returns
    -------
    dict, keyed by
        testPred : numpy.array 
            prediction on test set
            
        testErr : numpy.array
            test error over all bootstraps
            
        meanTestErr : float
            mean error over all bootstraps
            
        ID : str
            identifier
            
        y_test : numpy.array
            test target over all bootstraps
            
        weights : numpy.array
            feature importances
            
        testInd : numpy.array
            indix mask of test samples for each bootstrap
            
        trainErr : numpy.array
            training error over all bootstraps
        
    """
    
    # definitions and initialisations
    m, n_col = len(df), len(features)+1
    if data_func.is_iterable(m_test)==True:
        n_boot=1
    elif m_test==1: 
        n_boot=m # one fit for each observation
    if method=='VAR': 
        n_boot=m_test=1
    # empty fields for bootstrapped model output
    test_ref_Y,   test_pred_Y  = np.array([]), np.array([]) # test target values and out-of-sample predictions
    train_ref_Y,  train_pred_Y = np.array([]), np.array([]) # training target values and in-sample predictions
    train_error,  test_error   = np.array([]), np.array([]) # in and out-of-sample errors
    boot_errors,  models       = np.array([]), np.array([]) # mean bootstrap error and bootstrap models
    feat_weights, test_indices = np.zeros((n_boot,n_col-1)), np.zeros((n_boot,m)) # weights for feature importance, test_index over bootstraps
    
    # input data
    inputs = df.copy()
    if not to_norm==None: # normalise data (z-scores)
        for var in to_norm:
            if var in inputs.columns:
                vals        = inputs[var].values
                inputs[var] = (vals-vals.mean(0))/vals.std(0,ddof=1)
            else:
                raise ValueError("Norm error: Variable '{0}' not in dataframe.".format(var))
    
    # loop over bootstrapped samples
    for t in range(n_boot):
        # get training and testing data
        if data_func.is_iterable(m_test)==True:
            df_train, df_test = inputs[~m_test], inputs[m_test]
            test_indices[t,:] = m_test
        else:
            df_train, df_test, is_train = train_test_split(inputs,m_test=m_test,t=t) # random split
            test_indices[t,:]           = ~is_train
        # get values
        x_train, y_train = df_train[features].values, df_train[target].values
        x_test,  y_test  = df_test[features].values,  df_test[target].values
        
        # set learning methods
        if not method=='VAR': # VAR part of statsmodels library (treated differently)
            ML = model_selection(method,n_HN=n_col-1,CV_name=CV_name,CV_value=CV_value) # n_HN only used for neural network
                                                                            # (nNeurons=nFeatures in each layer)
        else: # can only be used with m_test==1
            input_data = inputs[[target]+features].values
            ML         = model_selection(method,input_data)
            y_train    = y_test = input_data[:,0]
            if CV_name==None: model = ML.fit(maxlags=1) # model fit, defaults VAR with one lag
            else:      exec('model = ML.fit('+CV_name+'='+str(CV_value)+')')
        
        # fit model and train/test predictions
        if method=='VAR': # fit at method selection step (CV_name needed)
            in_pred  = np.zeros(m)*np.nan
            for r in range(m):
                start_values = input_data[r,:]
                fcast        = model.forecast(start_values.reshape((1,len(features)+1)),horizon)[-1,0]
                if r+horizon<m:
                    in_pred[r+horizon]  = fcast
            out_pred = in_pred
        else:
            model_clone  = skl_base.clone(ML)
            model        = ML.fit(x_train,y_train) # model fit
            out_pred     = model.predict(x_test)
            in_pred      = model.predict(x_train)
        
        # get discrete class output & get bootstrap error
        if is_class==True: # target should be an integer
            if is_zero_one==True: # map to Boolean
                in_pred  = data_func.to_zero_one(in_pred).astype(bool)
                out_pred = data_func.to_zero_one(out_pred).astype(bool)
            else: # map to integer
                in_pred  = np.round(in_pred).astype(int)
                out_pred = np.round(out_pred).astype(int)
            boot_errors = np.hstack((boot_errors,np.mean(out_pred!=y_test)))
        else:
            if method=='VAR':
                boot_errors = np.nanmean(np.abs(out_pred-y_test))
            else:
                boot_errors = np.hstack((boot_errors,np.mean(np.abs(out_pred-y_test))))
        models = np.hstack((models,model)) # store model
        
        # feature importance
        if counter_fact==False:
            if method in ['Tree-rgr','Tree-clf','Forest-rgr','Forest-clf']:
                feat_weights[t] = model.feature_importances_
        # feature importance through "counter_factual" analysis (leave one variable out and compare)
        elif counter_fact==True: # may slow things down
            for f,feat in enumerate(features):
                model_clone_II = skl_base.clone(model_clone)
                temp_features = list(features)
                temp_features.remove(feat)
                # get training and testing data
                x_train, x_test = df_train[temp_features].values, df_test[temp_features].values
                temp_model      = model_clone_II.fit(x_train,y_train)
                temp_pred       = temp_model.predict(x_test)
                if is_class==True:
                    feat_weights[t,f] = np.mean(temp_pred!=y_test)
                else:
                    feat_weights[t,f] = np.mean(np.abs(temp_pred-y_test))
        # train Ys
        train_pred_Y = np.hstack((train_pred_Y, in_pred))
        train_ref_Y  = np.hstack((train_ref_Y,  y_train))
        # test Ys
        test_pred_Y  = np.hstack((test_pred_Y,  out_pred))
        test_ref_Y   = np.hstack((test_ref_Y,   y_test))
    
    # get errors    
    if is_class==True:
        train_error  = np.mean(train_pred_Y!=train_ref_Y)
        test_error   = np.mean(test_pred_Y!=test_ref_Y)
    else:
        train_error  = np.mean(np.abs(train_pred_Y-train_ref_Y))
        test_error   = np.mean(np.abs(test_pred_Y-test_ref_Y))
    
    # verbose
    ID = target+'-'+method+'-'+str(m_test)+'-'+str(n_boot)
    if verbose==True:
        print '\nTraining Summary'
        print 'ID:',ID
        print '\tin-sample error:',round(train_error,3)
        print '\tout-of-sample error:',round(test_error,3)
        print '\terror variance:',round(np.std(boot_errors,ddof=1),3)
        print '\terror signal-to-noise:',
        print round(test_error/np.std(boot_errors,ddof=1),3)
    
    # package output
    out_dict = {'ID' : ID,\
                'mean_train_err' : train_error,  'mean_test_err' : test_error,\
                'train_pred_Y'   : train_pred_Y, 'test_pred_Y'   : test_pred_Y,\
                'train_ref_Y'    : train_ref_Y,  'test_ref_Y'    : test_ref_Y,\
                'feat_weights'   : feat_weights, 'test_ind'      : test_indices}
    if save_models==True:
        out_dict['models']=np.array(models)
    if save_out==True:
        pk.dump(out_dict,open(file_name,'wb'))
    if save_models==False: # if not saved, keep models in temp (full) output
        out_dict['models']=np.array(models)
    
    # return output dictionary
    return out_dict
    

def train_test_split(df,m_test=None,t=0):
    """Randomly split data into training and test set.
    
       options: m_test=1    : jack-knifing (specify index t)
                    m_test=None : split by half
                    m_test=nbr  : sample 'nbr' observations
                    
        returns: 
            
            training dataframe, test dataframe, training index mask

    Parameters
    ----------
    df : numpy.array or pandas.DataFrame
        input dataset to be split
        
    m_test : value, optional (Default value = None)
        None: split in half
        int: number to be sampled for testing
        1: index of observation for testing (jack-knifing)
    t : int, optional (Default value = 0)
        index, if m_test==1

    Returns
    -------
    df_train : numpy.array or pandas.DataFrame
        sample training data
        
    df_test : numpy.array or pandas.DataFrame
        sampled test data
        
    is_train : numpy.array
        index mask of training data
        
    """
        
    m = len(df)
    
    # split sample in half
    if m_test==None:
        m_test=m/2
        train_i = rn.sample(range(m),m-m_test)
    
    # "jack-knifing" (select the t'th observation for testing)
    elif m_test==1:
        train_i = np.arange(m)
        train_i = np.delete(train_i,t)
    
    # random sample for training and testing (n_boot-times)
    else:
        train_i = rn.sample(range(m),m-m_test)
    is_train = np.in1d(range(m),train_i,assume_unique=True)
    df_train = df[is_train]
    df_test  = df[~is_train]
    
    return df_train, df_test, is_train
    

def model_selection(method,data=None,n_HN=None,CV_name=None,CV_value=None):
    """Select model instance from scikit-learn library.

    Parameters
    ----------
    method : str
        model type, options: NB' (Naive-Bayes),'SVM','NN' (neural net), 'Tree','Forest','kNN','Logit','OLS','VAR'
                             suffix: 'rgr' : regression problem, 'clf' : classification problem
                             NB only takes clf, No suffix for Logit, OLS and 'VAR'
   
   data : numpy.array, optional (Default value = None)
        needs to be given if method=='VAR' (from 'statsmodels', not 'scikit-learn').
        
    n_HN : int, optional (Default value = None)
        number of neurons in hidden layer (only needed for neural networks)
        
    CV_name : str, optional (Default value = None, no cross-validation)
        name of cross-validation parameter.
        Note: default and cross-validated model require to modify the source code below.
        
    CV_value : value, optional (Default value = None)
        value of CV_name, if not None.

    Returns
    -------
    model : scikit-learn model instance (VAR from statsmodels)

    """
    
    # check if model choice is valid
    valid_methods = ['NN-rgr','NN-clf','Tree-rgr','Tree-clf','Forest-rgr','Forest-clf',\
                     'SVM-rgr','SVM-clf','kNN-rgr','kNN-clf','NB-clf','OLS','Logit','VAR']
    if not method in valid_methods:
        raise ValueError("Invalid method: '{0}' not supported.".format(method))
    
    # select model
    else:
        if   method=='NN-rgr': # ONLY WORKING with scikit-learn >= 0.18
            # create and train network
            if CV_name==None: model = skl_nn.MLPRegressor(hidden_layer_sizes=(n_HN,n_HN),alpha=.001,activation='relu',solver='lbfgs')
            else:      exec('model = skl_nn.MLPRegressor('+CV_name+'='+str(CV_value)+\
                                        ',hidden_layer_sizes=((n_col-1),(n_col-1)),activation="relu",solver="lbfgs")')
        elif method=='NN-clf': # ONLY WORKING with scikit-learn >= 0.18
            # create and train network
            if CV_name==None: model = skl_nn.MLPClassifier(hidden_layer_sizes=n_HN,alpha=2.,activation='logistic',solver='lbfgs')
            else:      exec('model = skl_nn.MLPClassifier('+CV_name+'='+str(CV_value)+\
                                        ',hidden_layer_sizes=(n_col-1),activation="logistic",solver="lbfgs")')
        elif method=='Tree-rgr':
            if CV_name==None: model = skl_tree.DecisionTreeRegressor(max_features='sqrt',max_depth=5) 
            else:      exec('model = skl_tree.DecisionTreeRegressor('+CV_name+'='+str(CV_value)+',max_features="sqrt")')
        elif method=='Tree-clf':
            if CV_name==None: model = skl_tree.DecisionTreeClassifier(max_depth=8)
            else:      exec('model = skl_tree.DecisionTreeClassifier('+CV_name+'='+str(CV_value)+')') # e.g. 'max_depth'
        elif method=='Forest-rgr':
            if CV_name==None: model = skl_ens.RandomForestRegressor(200,max_features='sqrt',max_depth=11) # 7 best for CPI
            else:      exec('model = skl_ens.RandomForestRegressor(200,'+CV_name+'='+str(CV_value)+',max_depth=11)')
        elif method=='Forest-clf':
            if CV_name==None: model = skl_ens.RandomForestClassifier(200,max_depth=9,\
                                       criterion='entropy',max_features='sqrt')
            else:      exec('model = skl_ens.RandomForestClassifier(200,'+CV_name+'='+str(CV_value)+',criterion="entropy")')
        elif method=='SVM-rgr':
            if CV_name==None: model = skl_svm.SVR(C=50,gamma=0.001,epsilon=0.2)
            else:      exec('model = skl_svm.SVR('+CV_name+'='+str(CV_value)+',epsilon=0.2,gamma=0.001)') # change for gamma/epsilon CV
        elif method=='SVM-clf':
            if CV_name==None: model = skl_svm.SVC(C=1e3,gamma=1)
            else:      exec('model = skl_svm.SVC(C=1e3,'+CV_name+'='+str(CV_value)+')')        
        elif method=='kNN-rgr':
            if CV_name==None: model = skl_neigh.KNeighborsRegressor(n_neighbors=2,p=1)
            else:      exec('model = skl_neigh.KNeighborsRegressor('+CV_name+'='+str(CV_value)+',p=1)') # e.g. 'n_neighbors'
        elif method=='kNN-clf':
            if CV_name==None: model = skl_neigh.KNeighborsClassifier(n_neighbors=5,p=1)
            else:      exec('model = skl_neigh.KNeighborsClassifier(n_neighbors=3,'+CV_name+'='+str(CV_value)+')') # e.g. 'n_neighbors'        
        elif method=='NB-clf':
            if CV_name==None: model = skl_NB.GaussianNB()
            else:      exec('model = skl_NB.MultinomialNB('+CV_name+'='+str(CV_value)+')') # e.g. 'alpha' smoothing parameter (0 for no smoothing). 
        elif method=='OLS':
            if CV_name==None: model = skl_lin.Ridge(alpha=10)
            else:      exec('model = skl_lin.Ridge('+CV_name+'='+str(CV_value)+')')
        elif method=='Logit':
            if CV_name==None: model = skl_lin.logistic.LogisticRegression(C=0.01)
            else:      exec("model = skl_lin.logistic.LogisticRegression("+CV_name+"="+str(CV_value)+",solver='liblinear')")
        elif method=='VAR':
            model = sm_vars.VAR(data)
    
    return model


def eval_model_ensemble(models,x,y_ref=None,is_class=False,verbose=False):
    """Evaluate ensemble of models.
    
    Parameters
    ----------
    models : single or iterable set of scikit-learn model instances
        model(s) to be evaluated
        
    x : numpy.array
        model inputs with m-row observations and n-column features
        
    y_ref : numpy.array (Default value = None)
        reference target output for observations in X
        
    is_class: bool (Default value = False)
        indication if classification problem (only needed when Y!=None)
        
    Returns
    -------
    model output : numpy array of len(X)
        if Y!=None: list of length 2 [mean error to reference Y, model outputs]
    
    """
    
    # model evaluation
    if len(np.array(x).shape)==1: # single observation input
        if data_func.is_iterable(models)==False: # single model
            y_pred = models.predict(x)
        else:
            y_pred = np.zeros(len(models))  # multiple models
            for m,mo in enumerate(models):
                y_pred[m] = mo.predict(x)
    elif len(np.array(x).shape)==2:
        if data_func.is_iterable(models)==False: # single model
            y_pred = models.predict(x)
        else:
            y_pred = np.zeros((len(x),len(models))) 
            for m,mo in enumerate(models):
                y_pred[:,m] = mo.predict(x)
    else:
        raise ValueError('Feature imput dimension greater than 2.')
            
    # error evaluation
    if y_ref==None:
        return y_pred
    else:
        if is_class==False: # regression problem
            y_err = np.mean(np.abs(y_pred-y_ref))
        else: # classification problem
            y_err = np.mean(np.abs(y_pred!=y_ref))
        if verbose==True:
            print '\nMean model error: {0}.'.format(np.round(y_err,2))
        return [y_pred, y_err]
    
    
def prec_rec_F1(target,prediction,use_df=False,df=None,df_start=0,df_end=1,\
                T=True,digits=3,ID='',verbose=True):
    """Precision, recall and F1 scores for binary classification results.

    Parameters
    ----------
    target : numpy.array or pandas.DataFrame
        reference values
        
    prediction : numpy.array or pandas.DataFrame
        model values
        
    use_df : bool, optional (Default value = False)
        if True, expect pandas.DataFrame as inputs (target and predictions)
        
    df : pandas.DataFrame, optional (Default value = None)
        input data (target and predictions)
        
    df_start : value, optional (Default value = 0)
        index start value for evaluation
        
    df_end : value, optional (Default value = 1)
        index end value for evaluation
        
    T : bool, optional (Default value = True)
        if True, True is True, else False is True ;-)
        
    digits : int, optional (Default value = 3)
        number of digits to which results are rounded
        
    ID : str, optional (Default value = '')
        identifier
        
    verbose : bool, optional (Default value = True)
        if True, print results to screen.

    Returns
    -------
    prec : float
        precision score
        
    rec : float
        recall score
        
    F1 : float
        F1-score

    """
    
    if use_df==True:
        target     = df[df_start:df_end][target].values
        prediction = df[df_start:df_end][prediction].values
    
    # remove nan's
    is_OK      = ~np.isnan(target) & ~np.isnan(prediction)
    target     = target[is_OK]
    prediction = prediction[is_OK]
    
    # get values
    acc     = 100*np.sum(prediction==target)/float(len(target))
    true_pos  = float(np.sum((prediction==T) & (target==T)))
    false_pos = np.sum((prediction==T) & (target!=T))
    false_neg = np.sum((prediction!=T) & (target==T))
    if true_pos>0:
        prec    = 100*true_pos/(true_pos+false_pos)
        rec     = 100*true_pos/(true_pos+false_neg)
        F1      = 2*prec*rec/(prec+rec)
    else:
        prec,rec,F1 = 0.,0.,0.
    
    # print results
    if verbose==True:
        print '\nBinary Classification Stats (%)'
        print 'ID: {0}\n'.format(ID)
        print '\taccuracy  : {0}'.format(np.round(acc,  digits))
        print '\tprecision : {0}'.format(np.round(prec, digits))
        print '\trecall    : {0}'.format(np.round(rec,  digits))
        print '\tF-1 score : {0}.\n'.format(np.round(F1,   digits))
    return prec,rec,F1
    

def bias_var_sd_R2(target,prediction,use_df=False,df=None,\
                    df_start=0,df_end=1,digits=3,ID='',verbose=True):
    """calculate Bias, Variance, Std dev and R-squared for prediction.

    Parameters
    ----------
    target : numpy.array or pandas.DataFrame
        reference values
        
    prediction : numpy.array or pandas.DataFrame
        modelled values
        
    use_df : bool, optional (Default value = False)
        if True, expect pandas.DataFrame as inputs (target and predictions)
        
    df : pandas.DataFrame, optional (Default value = None)
        input data (target and predictions)
        
    df_start : value, optional (Default value = 0)
        index start value for evaluation
        
    df_end : value, optional (Default value = 1)
        index end value for evaluation
        
    digits : int, optional (Default value = 3)
        number of digits to which results are rounded
        
    ID : str, optional (Default value = '')
        identifier
        
    verbose : bool, optional (Default value = True)
        if True, print results to screen.

    Returns
    -------
    Bias : float
        mean error
        
    Var: float
        sample variance of bias
        
    Std : float
        sample standard deviation of bias
        
    R2: float
        r-squared score

    """    
    
    if use_df==True:
        target     = df[df_start:df_end][target].values
        prediction = df[df_start:df_end][prediction].values
    
    # remove nan's
    is_OK       = ~np.isnan(target) & ~np.isnan(prediction)
    target     = target[is_OK]
    prediction = prediction[is_OK]
    
    # get values
    Bias = np.mean(np.abs(prediction-target))
    Var  = np.var(np.abs(prediction-target),ddof=1)
    Std  = np.std(np.abs(prediction-target),ddof=1)
    Corr = st.pearsonr(prediction,target)[0]
    R2   = skl_metrics.r2_score(target,prediction)
    
    # print results
    if verbose==True:
        print '\nBias-Var-Std-R2 Stats'
        print 'ID:', ID
        print '\tBias      : {0}'.format(np.round(Bias,  digits))
        print '\tVariance  : {0}'.format(np.round(Var,   digits))
        print '\tStd. dev. : {0}'.format(np.round(Std,   digits))
        print '\tPearson-r : {0}'.format(np.round(Corr,  digits))
        print '\tR-squared : {0}.\n'.format(np.round(R2, digits))
    return Bias, Var, Std, R2
        

    
    
def k_folds(M,k=5):
    """Randomly extract k test folds from M observations
    
    Parameters
    ----------
    M : int
        number of observations in sample
        
    k : int, optional (Default value = 5)
        number of folds data should be separated in
        
    Returns
    -------
    test_ind : (M x k) numpy.array of type bool 
       index mask for k folds across M observations
    
    """
    
    M,k         = int(M), int(k) # insure integer type
    sample_size = M/k            # rounded  fold size
    fullset     = set(range(M))  # full index set
    test_ind    = np.zeros((M,k),dtype=bool) # final test index sets
    
    # loop over fold
    for i in range(k):
        if i < (k-1): # not last fold
            sample        = rdm.sample(fullset,sample_size)
            test_ind[:,i] = np.in1d(range(M),sample,assume_unique=True)
            fullset      -= set(sample)
        else: # catch the rest
            test_ind[:,i] = np.in1d(range(M),list(fullset),assume_unique=True)
            
    return test_ind
    
    
def get_feat_importance(W, max_norm=100.):
    """max-normed feature importance and its variance across models.

    Parameters
    ----------
    W : numpy.array
        input data
        
    max_norm : float, (Default value = 100)
        max norm
        
    Returns
    -------
    impo : number or array
        max-normed mean values (axis=0)
        
    error : number or array
        max-normed sample standard deviation (axis=0)
        
    """
    
    # mean variable importance and its standard deviation as measure of variation
    mean_imp,imp_norm,imp_sd =\
        np.nanmean(W,0),max(np.nanmean(W,0)),np.nanstd(W,0,ddof=1)
    impo  = max_norm*mean_imp/imp_norm
    error = max_norm*imp_sd/imp_norm
    
    return np.array([impo,error])
