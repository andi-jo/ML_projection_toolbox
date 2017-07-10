# -*- coding: utf-8 -*-
"""
PART D - DIAGNOSTIC PLOTS: Case study UK CPI inflation projections
------------------------------------------------------------------
from SWP XXX, Machine learning at central banks, Bank of England, 2017.
Authors: Chiranjit Chakraborty & Andreas Joseph.

See README.txt for details
"""

print '\n\nSeries of diagnostic plots'
print '--------------------------\n'

from __main__ import config,data,data_func,ml_func,ml_plot,plt,np

from __main__ import case # can be removed if no separated cases are given in the main file

if config.do_projections==True:
    from __main__ import pro
if config.do_shift_anal==True:
    from __main__ import sft


if case == 'UK_CPI':
    # color dict for target and features (plots), specify more colours as needed.
    color_seq  = ['k','#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
                     '#98df8a', '#8c564b', '#d62728', '#ff9896', '#9467bd', '#c5b0d5']
    color_dict = dict(zip([config.target]+list(config.features),color_seq))
    
    # data time series plot
    # ---------------------
    
    to_plot  = [config.target]+config.features[:-2] # exclude volatile components (ERI & Comm _idx)
    col_list = [color_dict[var] for var in to_plot]
    ref_time = list(data.data_no_shift.index.values).index(config.break_point) # GFC reference
    fig_name = config.fig_path+'Macro_timeseries_{0}.{1}'.format(data.ID_short,config.fig_format)
    
    # plot example
    p = data.data_no_shift[to_plot].plot(figsize=(9,6),color=col_list,lw=2)
    p.axhline(0,c='k',ls='--',label='zero reference')
    p.axvline(ref_time,ls='--',c='r',lw=2,label='GFC (break point)')
    p.set_ylabel('change over {0} {1} or level'.format(config.horizon,config.unit),fontsize=14)
    p.legend(bbox_to_anchor=(1.4, 1.02),prop={'size':14})
    if config.save_plots==True:
        plt.savefig(fig_name,dpi=200,bbox_inches='tight')
    plt.draw()
    
elif case=='BJ_air': # BJ air pollution example
    # color dict for target and features (plots), specify more colours if needed.
    color_seq     = ['k', 'm', 'c', '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
                     '#98df8a', '#8c564b', '#d62728', '#ff9896', '#9467bd', '#c5b0d5']
    color_dict     = dict(zip([config.target]+list(config.features),color_seq))


# projections analysis plots
# --------------------------
if config.do_projections==True:
    
    # projections.format(config.application,config.method)
    # -----------
    # data & settings
    df         = pro.results_dict['projections']
    fit_name   = 'mean fcast'
    title      = '{0} {1} with {2}% PI'.format(config.method,config.application,int(100-config.ptile_cutoff))
    fig_name   = config.fig_path+'ML_projections_{0}.{1}'.format(data.ID_short,config.fig_format)
    if case=='UK_CPI':
        GFC_impact = ['2011Q3','GFC_impact'] # for two year horizon
        ref_line   = [(1.02**(config.horizon/4.))*100-100,'target'] # annualised 2%-target
        y_range    = [-4,17] # approx y-value range for two year horizon
        x_label    = 'date'
        y_label    = '{0}-{1} change / %'.format(config.horizon,config.unit)

        # plot
        ml_plot.ML_projection_plot(df, fit_name, config.target, test_start=pro.end_times[0], test_end=pro.end_times[-1],\
                                   ref_col=config.ref_model, ref_time=GFC_impact, ref_level=ref_line,  y_lim=y_range,\
                                   pred_band = ['hi','lo','95% PI'], x_label=x_label, y_label= y_label, title=title,\
                                   save=config.save_plots,save_name=fig_name)
    
    elif case=='BJ_air':
        x_label    = 'time range ({0}) from 1 Jan 2010'.format(config.unit)
        y_label    = 'above/below 100pm threshold'
        start_hour = 39700
        end_hour   = 40050
        test_end   = 40000
        # plot
        ml_plot.ML_projection_plot(df.loc[start_hour:end_hour], fit_name, config.target, two_class=True, test_end=test_end,\
                                   x_label=x_label, y_label= y_label, title=title,save=config.save_plots,save_name=fig_name)
    
                       
    
    # conditional forecasts (heatmap)
    # -------------------------------
    # data & settings
    models   = pro.results_dict['models']
    f1, f2   = config.features[1],config.features[2] # variables (y,x)
    cond     = 'last' # condition: 'last','median', 'mean' or custom values
    title    = config.method+', condition: {0} ({1} {2} horizon)'.format(cond,config.horizon,config.unit)
    fig_name = config.fig_path+'ML_condition_{0}-{1}_{2}_{3}.{4}'.format(f1,f2,cond,data.ID_short,config.fig_format)
    # plot
    heat_vals = ml_plot.ML_heatmap(f1,f2,data.data_shifted,config.features,config.target,\
                                   models=models,ranges=None,condition=cond,N=30,to_norm=None,\
                                   title=title,save=config.save_plots,save_name=fig_name)
       
    # (un)conditioned fan-chart
    # -------------------------
    if case == 'UK_CPI':
        # data & settings
        ref_time, cond, models = '2015Q4', True, pro.results_dict['models']
        title   = '(un)conditioned fan chart for future projection'
        fig_name = config.fig_path+'fan_chart_{0}_ref-{1}-{2}.{3}'.format(data.ID_short,ref_time,cond,config.fig_format)
        
        # go one horizon length back and into the future  
        proj_dates       = pro.projections.index.values
        x_dates, y_dates = proj_dates[-3*config.horizon-1:-config.horizon], proj_dates[-2*config.horizon-1:]   
        df_X             = data.data_no_shift[config.features].loc[x_dates]
        df_X.index       = proj_dates[-2*config.horizon-1:] # shift index to prediction period
        df_Y             = data_func.data_framer(data.raw_data,config.target,config.features,index=config.time_var,\
                                                 start_i=config.start_time,end_i='2016Q4',\
                                                 shift=0,trafos=data.trafos,name_trafo=False).loc[y_dates]
        
        # plot
        ml_plot.cond_fan_chart(df_X,df_Y,models,ref_time,h_ref_line=ref_line,cond=cond,\
                               x_label=x_label,y_label=y_label,title=title,\
                               save=config.save_plots,save_name=fig_name)
                 
    # feature importance (full sample, fixed frequency, no time series)
    # -----------------------------------------------------------------
    if (config.counter_fact==True) or (config.method.split('-')[0] in ['Tree','Forest']):

        # data & settings
        importance, impo_sd = pro.results_dict['feat_imp'], pro.results_dict['feat_imp_sd']
        # target correlation with individual features
        target_feat_corr    = data.data_shifted[[config.target]+config.features].corr().values[1:,0]
        fig_name            = config.fig_path+'feature_importance_{0}.{1}'.format(data.ID_short,config.fig_format)
        
        # plot
        ml_plot.plot_feat_importance(importance,impo_sd,target_feat_corr,config.features,last=True,\
                                     color_dict=color_dict,save=config.save_plots,save_name=fig_name)

# plot horizon dependence of variable importance
# ----------------------------------------------
if config.do_shift_anal==True:

    # data & settings
    weights, variance = sft.feat_imp_dict['feat_imp'], sft.feat_imp_dict['feat_imp_sd']
    title             = 'variable importance: {0} ({1})'.format(config.application,config.method)
    fig_name          = config.fig_path+'horizon_dependence_{0}.{1}'.format(data.ID_short,config.fig_format)
    
    # plot
    ml_plot.plot_feat_shift_scores(weights,variance,color_dict,y_lim=[-10,120],title=title,\
                                   save=config.save_plots,save_name=fig_name)