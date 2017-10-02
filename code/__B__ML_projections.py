# -*- coding: utf-8 -*-
"""
PART B - PROJECTIONS: Case study UK CPI inflation projections
-------------------------------------------------------------
from Bank of England SWP 674: Machine learning at central banks (September 2017)
- authors:         Chiranjit Chakraborty & Andreas Joseph
- disclaimer:      licence.txt and SWP 674 disclaimer apply
- documentation:   see README.txt for structure and comments for details
"""

from __main__ import config,data_func,ml_func,data,np,pd,pk,time

from __main__ import case # can be removed if no separated cases are given in the main file

# print settings
print '\nProjections: time series training & testing:'
print '--------------------------------------------\n'

save_name  = config.out_path+'Projections_summary_'+data.ID_short # name used for in/output

# end times for sliding/expanding window
end_times  = data.data_shifted.index[config.init_train_period:]   # end of training periods (expanding or sliding window)
end_times  = end_times[np.arange(0,len(end_times),config.time_step_size)] # adjust for time step size

# check and correct for potential issue with the chosen end times
# ---------------------------------------------------------------
if not data.data_shifted.index[-1] in end_times: # force data end point to be included
    end_times = np.hstack((end_times,data.data_shifted.index[-1]))
    print '\tWarning: Final observation not included in series of training sets.'
    print '\t\tTraining period has been extended to include it\n'
if len(end_times)>1 and config.fixed_start==False:
    if config.init_train_period<10:
        print '\tWarning: Sliding window length has less than 20 observations.\n'
    if int(np.where(data.data_shifted.index==end_times[-1])[0]-\
           np.where(data.data_shifted.index==end_times[-2])[0])<19:
        print '\tWarning: Last  slice of sliding window has less than 20 observations.'
        print '\t\tLast slice will be merged with previous slice.\n'
        end_times = np.delete(end_times,int(np.where(end_times==end_times[-2])[0]))
elif len(end_times)==1 and config.fixed_start==False:
    print '\tWarning: Only one period for sliding window.\n'
L_end_time = len(end_times) # number of train/test intervals


# MODEL FIT or LOAD
# -----------------

if config.do_model_fit==True:
    start_time = time.time() # for time taking
    
    # initialisations for projections results
    col_name    = [config.target,'lo','mean fcast','hi','mean error',str(config.ref_model),str(config.ref_model)+' error']
    projections = np.zeros((data.M+config.horizon,len(col_name)))*np.nan
    projections[:data.M,0] = data.data_shifted[config.target].values
    
    # time index column for projections
    if config.time_var == 'rangeL':
        proj_index = np.arange(data.data_shifted.index[0],data.data_shifted.index[-1]+config.horizon+1)
    
    # >>>hard-coded<<< case
    elif case=='UK_CPI':
        # time range for projection period in quarters
        proj_index = pd.date_range(data.data_shifted.index[0], periods=data.M+config.horizon, freq='Q')
        proj_index = np.char.array(proj_index.year)+'Q'+np.char.array(proj_index.quarter)
    
    # initialisations for variable importance analysis       
    feat_imp    = np.zeros((L_end_time,len(config.features)))
    feat_imp_sd = np.zeros((L_end_time,len(config.features)))
    
    # LOOP over end times (expanding horizon)
    for t,end in enumerate(end_times):
        i_t     = int(np.where(proj_index==end)[0])
        i_t_hor = i_t+config.horizon
        
        # TRAINING
        # --------

        # start time
        if config.fixed_start==False and t>0:
            i_s   += config.time_step_size
            start  = data.data_shifted.index[i_s]
        else:
            i_s    = 0
            start  = data.data_shifted.index[i_s]
        
        # training data
        df_train = data_func.data_framer(data.data_shifted, config.target, config.features,\
                                         index=config.time_var, start_i=start, end_i=end, name_trafo=False)
        m_test_2 = int(np.round(len(df_train)*config.test_fraction)) # test data set fraction of total
        
        # model fit
        out_dict = ml_func.ML_train_tester(df_train, config.target, config.features, config.method,\
                                           is_class=config.is_class, m_test=m_test_2, n_boot=config.n_boot,\
                                           to_norm=config.to_norm, counter_fact=config.counter_fact,\
                                           verbose=config.verbose)
        # return model specification
        if t==0:
            print '\n\tModel specs:\n\n\t',out_dict['models'][0],'\n'
        
        # get variable importance
        p = ml_func.get_feat_importance(out_dict['feat_weights'])
        feat_imp[t,:], feat_imp_sd[t,:] = p[0], p[1]
        
        # VAR reference model
        if not config.ref_model==None:
            out_dict_ref = ml_func.ML_train_tester(df_train, config.target, config.features, method=config.ref_model,\
                                            is_class=config.is_class, horizon=config.horizon, m_test=m_test_2,\
                                            n_boot=config.n_boot, to_norm=config.to_norm, counter_fact=False,\
                                            verbose=config.verbose)
                                     
        # cross-validation fit for CV-values
        # ----------------------------------
        if not config.CV_name==None:
            # CV on initial training set
            if config.CV_at_last==False and t==0:
                do_CV = True
            # CV on last/full dataset
            elif config.CV_at_last==True and t==L_end_time-1:
                do_CV = True
            else: # no intermediate CV
                do_CV = False
            
            # cross-validation
            if do_CV==True:
                print '\n\tCross-validation (calibration) of "{0}" via "{1}":'.format(config.method,config.CV_name)
                print '\t\t{0} values between {1} and {2}.'.format(len(config.CV_values),config.CV_values[0],config.CV_values[-1])
                print '\t\tDo at last "{0}" over {1} observations.\n'.format(config.CV_at_last,len(df_train))
                
                X_CV     = df_train[config.features].values
                Y_CV     = df_train[config.target].values
                CV_error = np.zeros(len(config.CV_values))
                
                # loop over CV-vales
                for c,val in enumerate(config.CV_values):
                    CV_dict = ml_func.ML_train_tester(df_train.copy(), config.target, config.features, config.method,\
                                            is_class=config.is_class, m_test=m_test_2, n_boot=config.n_boot,\
                                            to_norm=config.to_norm, CV_name=config.CV_name, CV_value=val,\
                                            verbose=config.verbose)
                
                    # cross-validation model projection (out-of-bag)
                    CV_models = CV_dict['models']
                    CV_output = np.zeros((len(CV_models),len(X_CV)))*np.nan
                    # index correction: forest model saved as list of trees
                    if config.method.split('-')[0]=='Forest':
                        i_0 = int(len(CV_models)/config.n_boot)
                    else:
                        i_0 = 1
                    
                    test_I = CV_dict['test_ind'] # bootstrap index masks for text dataset
                    # predictions
                    for i,model in enumerate(CV_models):
                        CV_output[i,test_I[i/i_0]==1] = model.predict(X_CV[test_I[i/i_0]==1,:])
                    # errors
                    if config.is_class==False:
                        CV_error[c] = np.nanmean(np.abs(np.nanmean(CV_output,axis=0)-Y_CV))
                    else:
                        CV_error[c] = np.nanmean(np.round(np.nanmean(CV_output,axis=0))!=Y_CV)
            
                # format results andsave
                CVdf = pd.DataFrame({'CV values': config.CV_values, 'CV errors': CV_error})
                print '\n\tCross-Validation results: "{0}"\n\n{1}\n'.format(config.CV_name,CVdf)
                if config.save_results==True:
                    CV_save_name = config.out_path+'CV_summary_{0}.xlsx'.format(data.ID_short)
                    CVdf.to_excel(CV_save_name,sheet_name=config.method+', '+config.CV_name)
                                     
        # testing (3 parts)
        # -----------------
        models = out_dict['models']
        
        # VAR(1) reference model
        if not config.ref_model==None:
            VAR_model = out_dict_ref['models'][0]
        
        # PROJECTIONS PART I: out-of-bag till training end
        if t==0:
            train_L = len(df_train)
            # out-of-bag (pre-computed for bootstrapped models)
            X       = df_train[config.features].values
            Y       = projections[:train_L,0]
            foutput = np.zeros((len(models),len(df_train)))*np.nan
            # index correction: forest model saved as list of trees
            if config.method.split('-')[0]=='Forest':
                i_0 = int(len(models)/config.n_boot)
            else:
                i_0 = 1
            # loop over bootstrapped models
            test_I  = out_dict['test_ind'] # out-of-bag test indeces
            for i,model in enumerate(models):
                foutput[i,test_I[i/i_0]==1] = model.predict(X[test_I[i/i_0]==1,:])
            # record modelling stats
            # columns: inflation (0), low p-tile (1), mean forecast (2), high p-tile (3) 
            #          mean error (4), VAR reference (5), VAR error (6)
            projections[:train_L,1] = np.nanpercentile(foutput,config.ptile_cutoff/2,axis=0) # lower p-tile
            projections[:train_L,2] = np.nanmean(foutput,axis=0) # mean forecast
            projections[:train_L,3] = np.nanpercentile(foutput,100-config.ptile_cutoff/2,axis=0) # upper p-tile
            # mean point errors
            if config.is_class==False:
                projections[:train_L,4] = np.abs(np.nanmean(foutput,axis=0)-Y)
            else:
                projections[:train_L,4] = np.round(np.nanmean(foutput,axis=0))!=Y
            
            # VAR(1) reference
            if not config.ref_model==None:
                VAR_fcast = out_dict_ref['test_pred_Y']
                if config.is_class==False:
                    projections[:train_L,5] = VAR_fcast
                    projections[:train_L,6] = np.abs(projections[:train_L,5]-Y)
                else:
                    projections[:train_L,5] = np.round(VAR_fcast).astype(int)
                    projections[:train_L,6] = projections[:train_L,5]!=Y
                    
            # PROJECTIONS PART II: training end to horizon
            if not train_L==data.M:
                X     = data.data_shifted[config.features].iloc[train_L:train_L+config.horizon,:].values
                X_VAR = data.data_shifted[[config.target]+config.features].iloc[train_L-config.horizon:train_L,:].values
                Y     = projections[train_L:train_L+config.horizon,0] # target on horizon length
                for k in range(config.horizon):
                    foutput = np.zeros(len(models))*np.nan
                    for i,model in enumerate(models):
                        foutput[i] = model.predict(X[k].reshape(1,-1))
                    projections[train_L+k,1] = np.percentile(foutput,config.ptile_cutoff/2) 
                    projections[train_L+k,2] = np.mean(foutput)
                    projections[train_L+k,3] = np.percentile(foutput,100-config.ptile_cutoff/2)
                    if config.is_class==False:
                        projections[train_L+k,4] = np.abs(np.mean(foutput)-Y[k])
                    else:
                        projections[train_L+k,4] = np.round(np.mean(foutput))!=Y[k]
                
                    # VAR(1) reference
                    if not config.ref_model==None:
                        VAR_fcast = VAR_model.forecast(X_VAR[k].reshape((1,len(config.features)+1)),config.horizon)[-1,0]
                        if config.is_class==False:
                            projections[train_L+k,5] = VAR_fcast
                            projections[train_L+k,6] = np.abs(projections[train_L+k,5]-Y[k])
                        else:
                            projections[train_L+k,5] = np.round(VAR_fcast).astype(int)
                            projections[train_L+k,6] = projections[train_L+k,5]!=Y[k]
            
        # PROJECTIONS PART III: horizon projection till data end      
        else: # projections beyond initial training period
            # loop over length of time step
            for s in range(config.time_step_size):
                X       = data.data_no_shift[config.features].iloc[i_t-s,:].values # <<< CHECK
                X_VAR   = data.data_shifted[[config.target]+config.features].iloc[i_t-s,:].values
                Y       = projections[i_t_hor-s,0]
                foutput = np.zeros(len(models))*np.nan
                for i,model in enumerate(models):
                    foutput[i] = model.predict(X.reshape(1, -1))
                projections[i_t_hor-s,1] = np.percentile(foutput,config.ptile_cutoff/2) 
                projections[i_t_hor-s,2] = np.mean(foutput)
                projections[i_t_hor-s,3] = np.percentile(foutput,100-config.ptile_cutoff/2)
                if config.is_class==False:
                    projections[i_t_hor-s,4] = np.abs(np.mean(foutput)-Y)
                else:
                    projections[i_t_hor-s,4] = np.round(np.mean(foutput))!=Y
                    
                # VAR(1) reference
                if not config.ref_model==None:
                    VAR_fcast = VAR_model.forecast(X_VAR.reshape((1,len(config.features)+1)),config.horizon)[-1,0]
                    if config.is_class==False:
                        projections[i_t_hor-s,5] = VAR_fcast
                        projections[i_t_hor-s,6] = np.abs(projections[i_t_hor-s,5]-Y)
                    else:
                        projections[i_t_hor-s,5] = np.round(VAR_fcast).astype(int)
                        projections[i_t_hor-s,6] = projections[i_t_hor-s,5]!=Y
            
        # project beyond data end using fully trained model (Part 4)
        if config.full_model_proj==True and t==L_end_time-1:
            X       = data.data_no_shift[config.features].iloc[-config.horizon:,:].values
            foutput = np.zeros((len(models),config.horizon))*np.nan
            for i,model in enumerate(models):
                foutput[i] = model.predict(X)
            projections[-config.horizon:,1] = np.percentile(foutput,config.ptile_cutoff/2,axis=0) 
            projections[-config.horizon:,2] = np.mean(foutput,axis=0)
            projections[-config.horizon:,3] = np.percentile(foutput,100-config.ptile_cutoff/2,axis=0)
                     
        # modelling progress & summary stats
        fcast_time = proj_index[i_t_hor]
        fcast_val  = np.round(projections[i_t_hor,2],3)
        if np.isnan(fcast_val)==True:
            print '\n\tWarning: Time {0} not sampled for testing: return NaN value for projection.'.format(proj_index[i_t])
        if config.is_class==True:
            if not np.isnan(fcast_val)==True:
                fcast_val = int(fcast_val)
            if (i_t_hor) < data.M-1:
                y_val     = int(projections[i_t_hor,0])
                point_err = y_val==fcast_val
        else:
            if (i_t_hor) < data.M-1:
                y_val     = np.round(projections[i_t_hor,0],3)
                point_err = round(np.abs(y_val-fcast_val),3)
        
        mean_train_err = np.nanmean(np.abs(projections[:train_L,4]))
        mean_test_err  = np.nanmean(np.abs(projections[train_L:i_t_hor,4]))
        mean_ref_err   = np.nanmean(np.abs(projections[train_L:i_t_hor,6]))
        print "\tTraining {0}-{1}, error: {2}; Projection until {3}, model: {4}"\
                          .format(start,end,np.round(mean_train_err,3),fcast_time,fcast_val),
        if i_t_hor+1<=data.M: # error summary
            print 'actual: {0},\n\t\t\ttest errors: point ({1}), mean to horizon ({2})'.format(y_val,point_err,np.round(mean_test_err,3))
        if i_t_hor+1==data.M:
            print '\n\tFuture Projections:\n'
        elif i_t_hor+1>data.M:
            print
        
    # record results
    # --------------
    results_dict = {'ID': data.ID_long}
    results_dict['target']         = config.target, 
    results_dict['features']       = config.features
    
    # projections
    projections                    = pd.DataFrame(projections,columns=col_name)
    projections[config.time_var]   = proj_index
    projections.set_index(config.time_var, inplace=True)
    results_dict['projections']    = projections
    results_dict['mean_train_err'] = mean_train_err
    results_dict['mean_test_err']  = mean_test_err
    results_dict['mean_ref_err']   = mean_ref_err
             
    # variable importance
    feat_imp    = pd.DataFrame(feat_imp, columns=config.features,index=end_times)
    feat_imp_sd = pd.DataFrame(feat_imp_sd,columns=config.features,index=end_times)
    results_dict['feat_imp'], results_dict['feat_imp_sd'] = feat_imp, feat_imp_sd
    
    # cross-validation
    if not config.CV_name==None:
        results_dict['CV_summary'], results_dict['CV_error'] = CVdf, CV_error

    if config.save_models == True:
        results_dict['models'] = out_dict['models']
    
    # save results
    if config.save_results==True:
        pk.dump(results_dict,open(save_name+'.pkl','wb'))
        # write separate spreadsheet with projections
        projections.to_csv(config.out_path+'Projections_'+data.ID_short+'.csv')
    if config.save_models == False: # models not saved but contained in temp output
        results_dict['models'] = out_dict['models']
    
    # time and error summary
    totalT = np.round((time.time()-start_time)/60,2) # elapsed time in minutes
    print '\nDone in',totalT,'minutes.'
    print '\nModel errors'
    print '\tMean','{0: <11}'.format(config.method), 'test error:',np.round(mean_test_err,2)
    
    if not config.ref_model==None:
        print '\tMean','{0: <11}'.format(config.ref_model), 'test error:',np.round(mean_ref_err,2),'\n'
    if config.is_class==True:
        ml_func.prec_rec_F1(out_dict['test_ref_Y'],out_dict['test_pred_Y'],ID=data.ID_short)
        print '\tNote: OOB error for last training slice.'
        
# load pre-computed results
else:
    print 'Loading pre-computed results:\n\n\t',data.ID_long,'...',
    try:
        results_dict = pk.load(open(save_name+'.pkl','rb'))
        if not results_dict['ID']==data.ID_long:
            print '\n\tResults-ID not matching!'
        else:
            print '\ndone.\n'
    except IOError:
        print "Pre-computed results not found."
        

