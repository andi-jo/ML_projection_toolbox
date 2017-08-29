# -*- coding: utf-8 -*-
"""
Auxiliary plotting functions: Case study UK CPI inflation projections
---------------------------------------------------------------------
from Bank of England SWP 674: Machine learning at central banks (September 2017)
- authors:         Chiranjit Chakraborty & Andreas Joseph
- disclaimer:      licence.txt and SWP 674 disclaimer apply
- documentation:   see README.txt for structure and comments for details
"""

from __main__ import np,pd,plt,colors,cmx,patch


def ML_projection_plot(df,fit_col,target,test_start=None,test_end=None,ref_col=None,ref_time=None,ref_level=None,\
                       pred_band=None,idx=None,two_class=False,y_lim=None,x_label='',y_label='',title='',\
                       save=True,save_name='ML_projections_plot.png'):
    """Plot machine learning projections for lead-lag model.

    Parameters
    ----------
    df : pandas.DataFrame
        output from projection exercise
        
    fit_col : str
        name of model output column
        
    target : str
        name of target column
        
    test_start : value (Default value = None)
        (time) index value where training period ends and test period starts
        
    test_end : value (Default value = None)
        (time) index value where test period ends
        
    ref_name : str, optional (Default value = None)
        name of reference data column
        
    ref_time : [value,label], optional (Default value = None)
        (time) index value to mark special point and its label
        
    ref_level : [float,label], optional (Default value = None)
        y-value for horizontal reference line and its label
        
    pred_band : [upper_name,lower_name,label], optional (Default value = None)
        column names for upper and lower values of prediction intervals and label 
        
    idx : str, optional (Default value = None)
        name of index variable
        
    two_class : bool, optional (Default value = False)
        if True, draw 0.5 decision line for (0,1)-classification problem
        
    y_lim : [min_value,max_value], optional (Default value = None)
        y-boundaries of plot
        
    title : str, optional (Default value = '')
        plot title
        
    x_label : str, optional (Default value = '')
        plot y-axis label
        
    y_label : str, optional (Default value = '')
        plot y-axis label
        
    save : bool, optional (Default value = True)
        if True, save plot
        
    save_name : str, optional (Default value = 'ML_projections.png')
        file name under which to save plot (incl directory)
        
    Note: plot can be further adjusted by modifying code below.
    
    """
    
    # set index if not given
    if not idx==None:
        df.set_index(idx,inplace=True)
    
    # lines plots and confidence intervals
    p=df[[target,fit_col]].plot(figsize=(11,6),linewidth=2.5,\
           style=['bo-','gs-'],ms=3.5,rot=0) # main model output
    
    x0 = p.get_xlim()[0] # left boundary of x-axis (plot reference point)
    
    # prediction intervals 
    if not pred_band==None:
        p.fill_between(range(len(df)),df[pred_band[0]].values,df[pred_band[1]].values,color='r',alpha=.4)
        pi_fill = patch.Patch(color='r',alpha=.4)
    
    # reference fit
    if not ref_col==None:
        df[ref_col].plot(linewidth=2.5,style='kd-',ms=3.5,rot=0,alpha=.35,label=ref_col)
    
    # plot target and decision line
    if not ref_level==None:
        p.axhline(ref_level[0],ls='--',c='k',lw=5,alpha=.4,label=ref_level[1])
    if two_class==True:
        p.axhline(.5,ls='--',c='k',lw=2,alpha=.6,label='decision line')
        p.set_yticks([0,1])
    
    # indicate training and test periods
    if not test_start==None:
        t_s = x0+list(df.index.values).index(test_start)
        p.axvline(t_s,ls='--',c='k',lw=3,label='test start')
    else:
        t_s = x0
        
    if not test_end==None:
        t_e = x0+list(df.index.values).index(test_end)
        p.axvline(t_e,ls='-.',c='k',lw=3,label='test end')
    else:
        t_e = len(df)-1
    
    # hightlight special point in time
    if not ref_time==None:
        t_ref = x0+list(df.index.values).index(ref_time[0])
        p.axvline(t_ref,ls='-.',c='r',lw=3,label=ref_time[1])
    
    # error summaries
    box, fsize  = {'facecolor':'black', 'alpha':0.1, 'pad':12}, 15
    abs_fit_err = np.abs(df[target].values-df[fit_col].values)
    if not ref_col==None:
        abs_ref_err = np.abs(df[target].values-df[ref_col].values)
    
    # training period
    if t_s>x0:
        fit_train_err = np.round(np.nanmean(abs_fit_err[:t_s]),2)
        p.text(.20,.67,'out-of-bag error\n\n          '+str(fit_train_err), bbox=box,transform=p.transAxes, fontsize=fsize-3)
        if not ref_col==None:
            ref_train_err = np.round(np.nanmean(abs_ref_err[:t_s]),2)
            p.text(.20,.06,ref_col+' out-of-bag\n\n        '+str(ref_train_err),bbox=box,transform=p.transAxes, fontsize=fsize-5)
    
    # test period
    if not ref_time==None: # split up error start: before and after t_ref
        fit_err_1 = np.round(np.nanmean(abs_fit_err[t_s:t_ref]),2)
        fit_err_2 = np.round(np.nanmean(abs_fit_err[t_ref:t_e]),2)
        p.text(.45,.67,'test error (I)\n\n      '+str(fit_err_1),bbox=box,transform=p.transAxes, fontsize=fsize-3)
        p.text(.67,.67,'test error (II)\n\n      '+str(fit_err_2),bbox=box,transform=p.transAxes, fontsize=fsize-3)
        if not ref_col==None:
            ref_err_1 = np.round(np.nanmean(abs_ref_err[t_s:t_ref]),2)
            ref_err_2 = np.round(np.nanmean(abs_ref_err[t_ref:t_e]),2)
            p.text(.45,.06,ref_col+' error (I)\n\n      '+str(ref_err_1),bbox=box,transform=p.transAxes, fontsize=fsize-5)
            p.text(.67,.06,ref_col+' error (II)\n\n      '+str(ref_err_2),bbox=box,transform=p.transAxes, fontsize=fsize-5)
    else: # single error stats for test period
        fit_err = np.round(np.nanmean(abs_fit_err),2)
        p.text(.45,.67,'test error\n\n  '+str(fit_err),bbox=box,transform=p.transAxes, fontsize=fsize-3)
        if not ref_col==None:
            ref_err = np.round(np.nanmean(abs_ref_err),2)
            p.text(.45,.06,ref_col+' error\n\n  '+str(ref_err),bbox=box,transform=p.transAxes, fontsize=fsize-5)
    
    # labels, axes, legend
    p.set_xlabel(x_label,   fontsize =fsize)
    p.set_ylabel(y_label,   fontsize =fsize)
    p.set_title(title,      fontsize =fsize)
    p.tick_params(axis='x', labelsize=fsize-3)
    p.tick_params(axis='y', labelsize=fsize-3)
    if not y_lim==None:    
        p.set_ylim(y_lim)
    handles, labels = p.get_legend_handles_labels()
    if not pred_band==None:
        handles += [pi_fill]
        labels  += [pred_band[2]]
    p.legend(handles, labels, loc='upper right',ncol=4,prop={'size':fsize-2})
    
    # save figure
    if save==True:
        plt.savefig(save_name,dpi=200,bbox_inches='tight')
    plt.draw()
        

def cond_fan_chart(df_X,df_Y,models,ref_time,cond=True,idx=None,h_ref_line=None,data_return=False,\
                   two_class=False,legend_loc='best',y_lim=None,y_label=None,x_label=None,title='',\
                   save=False,save_name='cond_fan_chart.png'):
    """Percentile-based fan chart, optionally conditioned on Y-reference at reference time.

    Parameters
    ----------
    df_X : pandas.DataFrame
        input data for models
        
    df_Y : pandas.DataFrame
        
    models : list-like,
        fitted models
        
    ref_time : value
        index value of reference time
        
    cond : bool, optional (Default value = True)
        if True, force model mean on reference point
        
    idx : str, optional (Default value = None)
        name of index if not set
        
    h_ref_line : float, optional (Default value = None)
        y-value for horizontal reference line
        
    data_return : bool, optional (Default value = False)
        if True, return plot input data
        
    two_class : bool, optional (Default value = False)
        if True, two-class classification is assumed
        
    legend_loc : str or int, optional (Default value = 'best')
        matplotlib legend location    
    
    y_lim : [min_value,max_value], optional (Default value = None)
        y-boundaries of plot
        
    y_label : str, optional (Default value = None)
        y-axis label
        
    x_label : str, optional (Default value = None)
        x-axis label
         
    title : str, optional (Default value = '')
        plot title
        
    save : bool, optional (Default value = True)
        if True, save plot
        
    save_name : str, optional (Default value = 'cond_fan_chart.png')
        file name under which to save plot (incl directory)
        
    Note: plot can be further adjusted by modifying code below.

    Returns
    -------
    df : pandas.DataFrame
        internally generated data used for plot

    """
    
    # set index (df_X & df_Y need to have the same index)    
    if not idx==None:
        df_X.set_index(idx,inplace=True)
        df_Y.set_index(idx,inplace=True)
    
    # model input values based on X and models
    X = np.zeros((len(models),len(df_X)))
    for i,model in enumerate(models):
        X[i,:] = model.predict(df_X)

    # mean and percentiles: conditioned on reference point
    df, refY, ref_name = df_X.copy(), df_Y.loc[ref_time][df_Y.columns[0]], df_Y.columns[0]
    df['mean model'], df['median model']  = np.mean(X,axis=0), np.percentile(X,50,axis=0)
    mean_off, median_off = df.loc[ref_time]['mean model']-refY, df.loc[ref_time]['median model']-refY
    if cond==False:
        df['p25'],  df['p75']   = np.percentile(X,25,axis=0), np.percentile(X,75,axis=0)
        df['p5'],   df['p95']   = np.percentile(X,5,axis=0),  np.percentile(X,95,axis=0)
        df['p0.5'], df['p99.5'] = np.percentile(X,1,axis=0),  np.percentile(X,99,axis=0)
    else:
        df['mean model'], df['median model']  = df['mean model']-mean_off, df['median model']-median_off
        df['p25'],  df['p75']   = np.percentile(X,25,axis=0)-median_off,   np.percentile(X,75,axis=0)-median_off
        df['p5'],   df['p95']   = np.percentile(X,5,axis=0)-median_off,    np.percentile(X,95,axis=0)-median_off
        df['p0.5'], df['p99.5'] = np.percentile(X,1,axis=0)-median_off,    np.percentile(X,99,axis=0)-median_off
    # merge df and df_Y
    df = pd.concat([df_Y, df], axis=1)

    # plotting
    p=df[[ref_name,'mean model','median model']].plot(figsize=(9,6),linewidth=3,\
          style=['bo-','gs-','rd-'],ms=5,rot=0,alpha=.7)
    
    # reference
    ref_T=list(df.index.values).index(ref_time)
    p.axvline(ref_T,ls='--',c='k',lw=2)
    p.plot([ref_T], [refY], 'o', markersize=15, color='k',alpha=.5,label='ref.: '+str(ref_time))
    p.fill_between(range(len(df)),df['p25'].values,df['p75'].values,color='r',alpha=.2)
    r50=patch.Patch(color='r',alpha=.6)
    p.fill_between(range(len(df)),df['p5'].values,df['p95'].values,color='r',alpha=.2)
    r90=patch.Patch(color='r',alpha=.4)
    p.fill_between(range(len(df)),df['p0.5'].values,df['p99.5'].values,color='r',alpha=.2)
    r99=patch.Patch(color='r',alpha=.2)
    
    # add boundaries for two-class classification
    if two_class==True:
        p.axhline(0,ls='-',c='k',lw=.4)
        p.axhline(1,ls='-',c='k',lw=.4)
        if not y_lim==None:
            p.set_ylim(y_lim)
        else:
            p.set_ylim([-.25,1.5])  
        p.set_yticks([0,1])
    
    # add reference line and adjust legend ordering
    if not h_ref_line==None:
        p.axhline(h_ref_line[0],ls='-',c='k',lw=3,alpha=.3,label=h_ref_line[1])
        new_index = [0,5,3,1,6,4,2,7] # for legend ordering
    else:
        new_index = [0,4,3,1,5,2,6]
    
    # legend
    fsize = 15
    handles, labels = p.get_legend_handles_labels()
    handles += [r50,r90,r99]
    labels  += ['p-50','p-90','p-99']
    handles  = np.array(handles)[new_index]
    labels   = np.array(labels)[new_index]
    p.legend(handles,labels,loc=legend_loc,ncol=3,prop={'size':fsize-2},numpoints=1)
    
    # axes $ labels
    if not y_lim==None:    
        p.set_ylim(y_lim)
    if not y_label==None:
        p.set_ylabel(y_label,fontsize=fsize)
    if not x_label==None:
        p.set_xlabel(x_label,fontsize=fsize)
    p.set_title(title,fontsize=fsize)
    p.tick_params(axis='x', labelsize=fsize-2)
    p.tick_params(axis='y', labelsize=fsize-2)
    
    # save figure 
    if save==True:
        plt.savefig(save_name,dpi=200,bbox_inches='tight')
    plt.draw()
    
    # return underlying data   
    if data_return==True:
        return df

    

def ML_heatmap(f1,f2,df,features,target,models=None,model_outputs=None,condition='median',\
               N=30,ranges=None,to_norm=None,color_norms=None,title='',\
               color_map='rainbow',save=False,save_name='ml_heatmap.png'):
    """Heatmap of conditional 2-D model prediction.

    Parameters
    ----------
    f1 : str
        name of first variable feature
        
    f2 : str
        name of second variable feature
        
    df : pandas.DataFrame
        input data
        
    features : list of str
        names of model features (RHS)
        
    target : str
        name of target variables (LHS)
        
    models : list-like, optional (Default value = None)
        models to be evaluated. If None, needs pre-computed model_outputs

    model_outputs : 2-d numpy.array (NxN), optional (Default value = None)
        pre-computed model_outputs for f1-f2 feature ranges and condition
        
    condition : str or values, optional (Default value = 'median')
        condition for non-variable features, options: median, mean, last or custom values
            
    N : int, optional (Default value = 30)
        raster density within ranges
        
    ranges : [f1_min,f1_max,f2_min,f2_max], optional (Default value = None)
        ranges of variable features
        
    to_norm : list of str, optional (Default value = None)
        variable names to be normalised (z-scores)
     
    color_norms : [vmin,vmax], optional (Default value = None)
        range to norm color scale
        
    title : str, optional (Default value = '')
        plot title
        
    color_map : str, optional (Default value = 'rainbow')
        colormap, see also https://matplotlib.org/examples/color/colormaps_reference.html
        
    save : bool, optional (Default value = True)
        if True, save plot
        
    save_name : str, optional (Default value = 'ml_heatmap.png')
        file name under which to save plot (incl directory)
        
        
    Note: plot can be further adjusted by modifying code below.
    
    Returns
    -------
    df : 2-d numpy.array (NxN)
        heatmap values

    """
    
    data = df.copy()
    # normalise input data
    if not to_norm==None:
        for var in to_norm:
            vals      = data[var].values
            data[var] = (vals-vals.mean(0))/vals.std(0,ddof=1)
    df1f2 = [min(data[f1]),max(data[f1]),min(data[f2]),max(data[f2])]
    if condition=='median':
        inputs = data[features].median().values.reshape(1, -1)
        z      = data[target].median()
    elif condition=='mean':
        inputs = data[features].mean().values.reshape(1, -1)
        z      = data[target].mean()
    elif condition=='last':
        inputs = data[features].values[-1,:].reshape(1, -1)
        z      = data[target].values[-1]
    elif type(condition)==int:
        inputs = data[features].values[condition,:].reshape(1, -1)
        z      = data[target].values[condition]
    elif len(condition)==len(features):
        inputs = np.array(condition[1:]).reshape(1, -1)
        z      = condition[0]
    else:
        raise(ValueError('No valid modelling condition given.'))
    if ranges==None:
        ranges = df1f2
    elif not len(ranges)==4:
        raise(ValueError('Invalid feature ranges.'))
    # model prediction for models and feature ranges
    i1, i2 = features.index(f1), features.index(f2)
    y0, x0 = inputs[0][i1], inputs[0][i2]
    range1 = np.linspace(ranges[0],ranges[1],N)
    range2 = np.linspace(ranges[2],ranges[3],N)
    if model_outputs==None:
        output = np.zeros((len(models),N,N))
        for m,model in enumerate(models):
            for i,val1 in enumerate(range1):
                inputs[0,i1] = val1
                for j,val2 in enumerate(range2):
                    inputs[0,i2] = val2
                    output[m,i,j] = model.predict(inputs)
        output = np.mean(output[:,:,:],0) # model mean
    else:
        output = model_outputs
    # figure parameters
    if color_norms==None:
        vals = output.flatten()
        vmin = min(vals)
        vmax = max(vals)
    elif len(color_norms)==2:
        vmin,vmax=color_norms
    else:
        raise(ValueError('Invalid color norm.'))
    # plot
    fig, ax = plt.subplots(figsize=(8,6))
    # color map
    CMAP      = cm = plt.get_cmap(color_map)
    cNorm     = colors.Normalize(vmin=vmin, vmax=vmax)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=CMAP)
    im = ax.imshow(output,origin='lower',cmap=color_map,vmin=vmin,vmax=vmax,interpolation='hermite')
    ax.autoscale(False)
    
    # conditioning reference point
    x1 = (x0-ranges[2])*N/(ranges[3]-ranges[2])-.5
    y1 = (y0-ranges[0])*N/(ranges[1]-ranges[0])-.5
    ax.plot(x1,y1,'wo',ms=20)
    # condition point
    COL       = colorVal = scalarMap.to_rgba(z)
    ax.plot(x1,y1,'o',c=COL,ms=20,markeredgecolor='w',mew=3)
 
    fsize = 15 # figure base fontsize
    plt.title(title,fontsize=fsize)
    plt.xlabel(f2,  fontsize=fsize)
    plt.ylabel(f1,  fontsize=fsize)
    #tix = [0,int((N-1)/4),int((N-1)/2),int(3*(N-1)/4),N-1]
    tix = [0,int((N-1)/4),int((N-1)/2),int(3*(N-1)/4),N-1]
    plt.xticks(tix,np.round(range2[tix],1),fontsize=fsize-2)
    plt.yticks(tix,np.round(range1[tix],1),fontsize=fsize-2)
    cbar = plt.colorbar(im)
    cbar.set_label(target,fontsize=fsize)
    if save==True:
        plt.savefig(save_name,dpi=200,bbox_inches='tight')
    plt.draw()
    
    return output


def plot_feat_importance(weights,variance=None,corrs=None,features=None,last=False,\
                         y_label=None,x_mark=None,x_mark_label='',title='',color_dict=None,\
                         y_mark=None,y_lim=None,color_map='rainbow',\
                         save=False,save_name='feature_importance.png'):
    
    """Plot feature importance: time series or last.

    Parameters
    ----------
    weights : pandas.DataFrame
        feature importance scores
        
    variance : array, optional (Default value = None)
        error bands of feature importance scores
        
    corrs : array, optional (Default value = None)
        correlation between features and target
        
    features : list of str, optional (Default value = None)
        names of features
        
    last : bool, optional (Default value = False)
        if True, horizontal bar-chart of feature importance, else time series
        
    y_label : str, optional (Default value = None)
        y-axis label
        
    x_mark : value, optional (Default value = None)
        index value for x-axis reference value
        
    x_mark_label : str, optional (Default value = '')
        label of x-axes reference
        
    title : str, optional (Default value = '')
        plot title 
        
    color_dict : dict, optional (Default value = None)
        dictionary keyed by features and values providing color (if last==
        False)
        
    y_mark : values, optional (Default value = None)
         index value for y-axis reference value
         
    y_lim : [min_value,max_value], optional (Default value = None)
        y-boundaries of plot
        
    color_map : str, optional (Default value = 'rainbow')
        colormap, see also https://matplotlib.org/examples/color/colormaps_reference.html
        
    save : bool, optional (Default value = True)
        if True, save plot
        
    save_name : str, optional (Default value = 'feature_importance.png')
        file name under which to save plot (incl directory)
        
    Note: plot can be further adjusted by modifying code below.

    """
    
    fsize=15 # reference fontsize
    if features==None:
            features = weights.columns  
    if last==False: # plot time series
        if color_dict==None:
            fig = weights[features].plot(figsize=(8.5,6),lw=2,rot=30)
        else:
            color_seq = [color_dict[f] for f in features]
            fig = weights[features].plot(figsize=(8.5,6),color=color_seq,lw=2,rot=30)
        if not x_mark==None:
            x_mark = list(weights.index).index(x_mark)
            plt.axvline(x_mark,ls='--',lw=2,c='k',label=x_mark_label)
        if not y_mark==None:
            plt.axvline(y_mark,ls='-',lw=1,c='k')
        lgd = fig.legend(bbox_to_anchor=(1.4, 1.02),prop={'size':fsize-1})
        fig.tick_params(axis='x', labelsize=fsize-2)
        fig.tick_params(axis='y', labelsize=fsize-2)
        if not y_lim==None:    
            axes = plt.gca()
            axes.set_ylim(y_lim)
        if y_label==None:
            plt.ylabel('max-normed feature importance',fontsize=fsize)
        else:
            plt.ylabel(y_label,fontsize=fsize)
        plt.xlabel('date',fontsize=fsize)
        plt.title(title)
        if save==True:
            plt.savefig(save_name,dpi=200,bbox_extra_artists=(lgd,),bbox_inches='tight')
    else:
        # get feature importance and order values largest first
        if type(weights)==pd.core.frame.DataFrame:        
            impo = weights.values[-1,:]
        else:
            impo = weights
        order = impo.argsort()
        ranks = order.argsort()
        if type(variance)==pd.core.frame.DataFrame:
            error = variance.values[-1,:]
        else:
            error = variance
        error = error[order]
        fig, ax = plt.subplots(figsize=(8.5,6))
        # get correlation color
        if not corrs==None:
            CMAP      = cm = plt.get_cmap(color_map) 
            cNorm     = colors.Normalize(vmin=-1, vmax=1)
            scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=color_map)
            COL       = colorVal = scalarMap.to_rgba(corrs)
            ax.barh(ranks, impo, xerr=error, color=COL,align='center')
            scalarMap.set_array([-1,1])
            cb = fig.colorbar(scalarMap,ax=ax,ticks=np.arange(-1,1.1,.5))
            cb.set_label('target-feature correlation',rotation=270,fontsize=fsize-2)
        else:
            ax.barh(ranks, impo, xerr=error, color='r',align='center', alpha=0.4)
            xl = ax.get_xlim()
            if xl[1]>97:
                ax.set_xlim([0,110])
        # axes & ticks
        plt.yticks(ranks, features,fontsize=fsize-2)
        plt.axvline(100,ls='--',lw=0.5,color='k')
        plt.xlabel('max-normed feature importance',fontsize=fsize)
        axes = plt.gca()
        axes.set_xlim(left=0)
        axes.set_ylim([-1,len(features)])
        plt.title(title,fontsize=fsize)
    if save==True:
        plt.savefig(save_name,dpi=200,bbox_inches='tight')
    plt.draw()


def plot_feat_shift_scores(weights,variance,color_dict,y_lim=None,unit='',\
                           title='',save=True,save_name='test.png'):

    """Plot feature importance for different projection lead-lag length (horizons).

    Parameters
    ----------
    weights : pandas.DataFrame
        values of feature importance for difference horizons
        
    variance : pandas.DataFrame
        values of feature importance variance for difference horizons
        
    color_dict : dict, optional (Default value = None)
        dictionary keyed by features and values providing color
        
    title : str, optional (Default value = '')
        plot title
         
    y_lim : [min_value,max_value], optional (Default value = None)
        y-boundaries of plot
        
    save : bool, optional (Default value = True)
        if True, save plot
        
    save_name : str, optional (Default value = 'feature_importance.png')
        file name under which to save plot (incl directory)
        
    Note: plot can be further adjusted by modifying code below.

    """
    
    # get data
    imp_values   = weights.values
    imp_variance = variance.values
    names        = weights.columns
    times        = weights.index.values
    # plot
    fig, fsize = plt.figure(figsize=(8.5,6)), 15
    plt.axhline(100,lw=1.5,ls=':',c='k',label='max')
    for n,feat in enumerate(names):
        values = imp_values[:,n]
        hi,lo  = values+imp_variance[:,n], values-imp_variance[:,n]
        col    = color_dict[feat]
        plt.plot(times,values,lw=2,c=col,label=feat)
        plt.fill_between(times,lo,hi,color=col,alpha=.2)
    plt.axhline(0,lw=1,ls='--',c='k',label='min')
    plt.tick_params(axis='x', labelsize=fsize-2)
    plt.tick_params(axis='y', labelsize=fsize-2)
    plt.ylabel('max-normed feature importance',fontsize=fsize)
    if not unit=='':
        plt.xlabel('lead-lag length ({0})'.format(unit),fontsize=fsize)
    else:
        plt.xlabel('lead-lag length',fontsize=fsize)
    plt.xlim([min(times),max(times)])
    plt.title(title,fontsize=fsize)
    lgd = plt.legend(bbox_to_anchor=(1.37, 1.02),prop={'size':fsize-2})
    if not y_lim==None:    
        axes = plt.gca()
        axes.set_ylim(y_lim)
    # save figure
    if save==True:
        plt.savefig(save_name,dpi=200,bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.draw()
    
    

