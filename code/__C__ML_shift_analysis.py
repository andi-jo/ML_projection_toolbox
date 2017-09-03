# -*- coding: utf-8 -*-
"""
PART C - HORIZON ANALYSIS: Case study UK CPI inflation projections
------------------------------------------------------------------
from Bank of England SWP 674: Machine learning at central banks (September 2017)
- authors:         Chiranjit Chakraborty & Andreas Joseph
- disclaimer:      licence.txt and SWP 674 disclaimer apply
- documentation:   see README.txt for structure and comments for details
"""

from __main__ import config,data_func,ml_func,data,pd,np,time,pk

save_name = config.out_path+'Horizon_analysis_summary_'+data.ID_short

print '\nFeature importance for different lead-lag relations:'
print '----------------------------------------------------\n'
if config.do_model_fit==True:
    start_T = time.time() # for time taking
    
    # initialisations for results    
    feat_imp, feat_imp_sd, target_feat_corr =\
        np.zeros((len(config.time_shifts),\
        len(config.features))),np.zeros((len(config.time_shifts),len(config.features))),\
        np.zeros((len(config.time_shifts),len(config.features)))
    
    # loop over horizon lengths (shift values)
    for t,hor in enumerate(config.time_shifts):
        
        df_train = data_func.data_framer(data.raw_data.copy(),config.target,config.features,config.time_var,\
                                         config.start_time,config.end_time,shift=hor,\
                                         trafos=data.trafos,name_trafo=False)
        m_test_2 = int(np.round(len(df_train)*config.test_fraction)) # training set size        
        
        # model fit
        out_dict = ml_func.ML_train_tester(df_train,config.target,config.features,config.method,\
                                           is_class=config.is_class,m_test=m_test_2,n_boot=config.n_boot,\
                                           to_norm=config.to_norm,counter_fact=config.counter_fact,\
                                           verbose=config.verbose)
        
        # print model specification and progress
        if t==0:
            print '\tModel specs:\n\n\t',out_dict['models'][0],'\n'
            print '\tHorizon ({0}):'.format(config.unit),
        print hor,'..',
        
        # feature importance & target feature correlations
        p = ml_func.get_feat_importance(out_dict['feat_weights'])
        feat_imp[t,:], feat_imp_sd[t,:] = p[0], p[1]
        target_feat_corr[t,:] = df_train[[config.target]+config.features].corr().values[1:,0]
    
    # package results
    feat_imp         = pd.DataFrame(feat_imp,        columns=config.features,index=config.time_shifts)
    feat_imp_sd      = pd.DataFrame(feat_imp_sd,     columns=config.features,index=config.time_shifts)
    target_feat_corr = pd.DataFrame(target_feat_corr,columns=config.features,index=config.time_shifts)
    feat_imp_dict    = {'feat_imp':feat_imp, 'feat_imp_sd':feat_imp_sd,\
                        'target_feat_corr':target_feat_corr,'ID': data.ID_long}
    
    # save results
    if config.save_results==True:
        pk.dump(feat_imp_dict,open(save_name+'.pkl','wb'))
    totalT = np.round((time.time()-start_T)/60,2) # elapsed time in minutes
    print  'done.\n\nTime elapsed:',totalT,'minutes.\n\n'

# load pre-computed results
else:
    print '\nLoading pre-computed results:\n\t',data.ID_long,'...',
    try:
        feat_imp_dict= pk.load(open(save_name+'.pkl','rb'))
        if not feat_imp_dict['ID']==data.ID_long:
            print '\n\tResults-ID not matching!'
        else:
            print '\ndone.\n'
    except IOError:
        print "Pre-computed results not found."
