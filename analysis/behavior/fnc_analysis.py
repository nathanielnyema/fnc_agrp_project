import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
from scipy import stats as st
from statsmodels.stats.multitest import multipletests
from patsy import dmatrices
import statsmodels.formula.api as smf
import statsmodels.api as sm



#############################
## FUNCTIONS TO LOAD DATA ##
#############################

def load_data(fpath):
    """
    read the data from te specified file into a pandas dataframe
    """
    df = pd.read_csv(fpath)
    df = df.set_index(['Chr2','US', 'mouse', 'phase', 'day','CS']).sort_index()
    return df


################################
## FUNCTIONS TO REFORMAT DATA ##
################################

def subset_df(df, us, phase):
    """
    subset the dataframe loaded in load_data on just the data
    for a particular unconditioned stimulus ('glucose' or 'fat')
    and a particular phase of the experiment ('training' or 'test')
    """
    return df.loc[:,us,:,phase]


def average_test_data(df, Chr2, values = 'total_licks'):
    """
    subset the total testing licks dataframe on either the stim or control
    mice and average CS+ and CS- licks separately accross testing days
    
    the resulting output is the the format needed for most of the plotting functions below
    """
    return df.loc[Chr2,].pivot_table(index=['mouse','CS'], values = values)



def lick_microstructure(dft, dur, thresh = 0.5):
    
    dtg = dft.reset_index().set_index(['Chr2','sex','mouse', 'day', 'CS', 'lick_number']).time
    dtg = dtg - dtg.loc[:,:,:,:,:,0]
    dtg = dtg.loc[dtg<(dur*60)]

    def get_ili(x):
        y = pd.Series(x.values[1:] - x.values[:-1], 
                      index = np.arange(1,len(x), dtype = np.float64))
        y.index.name = 'lick_number'
        return y

    def label_bursts(x):
        x.loc[x.bs, 'bn'] = np.arange(x.bs.sum())
        return x
    
    ilis = dtg.groupby(['Chr2','sex','mouse', 'day', 'CS'], group_keys=True).apply(get_ili)
    
    dtg = pd.DataFrame(dtg)
    
    dtg['ilis'] = ilis
    dtg['ilis'] = dtg.ilis.fillna(thresh+1)
    dtg['bs'] = dtg.ilis > thresh
    dtg = dtg.groupby(['Chr2', 'sex', 'mouse', 'day', 'CS'], group_keys=True).apply(label_bursts)
    dtg['bn'] = dtg.bn.fillna(method = 'ffill')
    
    burst_sizes = dtg.groupby(['Chr2', 'sex', 'mouse', 'day', 'CS', 'bn']).size()
    get_burst_lens =  lambda x : x['time'].values[-1] - x['time'].values[0]
    burst_lens = dtg.groupby(['Chr2', 'sex', 'mouse', 'day', 'CS', 'bn'], group_keys=True).apply(get_burst_lens)
    bursts = pd.concat({'sizes': burst_sizes, 'lens': burst_lens}, axis = 1)
    bursts = bursts.loc[bursts.lens > 0]
    bursts['lr'] = bursts.sizes/bursts.lens
    
    burst_summ = bursts.groupby(['Chr2', 'sex', 'mouse', 'day', 'CS']).mean()
    burst_summ['tot'] = bursts.groupby(['Chr2','sex', 'mouse','day', 'CS']).sizes.sum()
    burst_summ['burst_num'] = bursts.groupby(['Chr2','sex','mouse','day','CS']).size()
    return bursts, burst_summ




############################
## FUNCTIONS FOR PLOTTING ##
############################

def two_bottle_plot(s, c, ax = None, mt_method = "holm-sidak", palette ='magma', lw = 1.5, ms = 5, y = 'total_licks',
                    plot_sig = True, alpha = .3, all_paired = False, groups=[r'$AgRP^{Chr2}$', r'$AgRP^{TdTomato}$'], t=30):
    """
    box plot comparing total average testing day licks for stim and ctl
    
    Parameters:
    -----------
    s: pd.DataFrame
        the testing dataframe subset on the "stim" group
    c: pd.DataFrame
        the testing dataframe subset on the "control" group
    """

    if ax is None:
        _, ax = plt.subplots(1,1)
    if isinstance(palette, list):
        palette = palette[::-1]
    s, c = s.copy(), c.copy()
    svc = pd.concat({groups[0]:s, groups[1]: c}, 
                    axis=0, names=['Condition']).reset_index()
    
    g = sns.barplot(data=svc, x = 'Condition', y= y, saturation = 1,
                hue = 'CS', hue_order = ['-', '+'], order = groups[::-1], 
                errorbar=None, ax = ax, palette = palette)
    handles, _ = g.get_legend_handles_labels()
    
    check_bar = lambda x: isinstance(x, mpl.patches.Rectangle)
    bars = list(filter( check_bar, g.get_children()))
    bars = list(filter( lambda x: sum(x.get_facecolor()) != 4, bars))
    locs = sorted([b.get_x() + b.get_width()/2 for b in bars])
    
    c['x'] = c.index.get_level_values('CS').map(lambda x: locs[0] if x == '-' else locs[1])
    s['x'] = s.index.get_level_values('CS').map(lambda x: locs[2] if x == '-' else locs[3])
    
    s.groupby('mouse').plot.line(x='x', y= y, marker = 'o', 
                                 alpha = alpha, lw = lw, color = 'k',
                                 ax = g, legend = False, ms = ms);
    c.groupby('mouse').plot.line(x='x', y= y, marker = 'o', 
                                 alpha = alpha, lw = lw, color = 'k', 
                                 ax = g, legend = False, ms = ms);
    g.legend(handles, ['CS -', 'CS+'], loc='lower left', bbox_to_anchor=(0,1), frameon=False) 

    sns.despine()
    
    stats = two_bottle_stats(s[y], c[y], mt_method = mt_method, groups = groups, all_paired = all_paired)
    [p_csp_csm_stim, p_csp_csm_ctl, p_csp_csp, p_csm_csm] = stats.pvalue_corr.values

    if plot_sig:
        #plot the results of the hypothesis testing
        y=max(s[y].max(), c[y].max())
        plot_significance(p_csp_csm_stim, ax, x1=.8, x2=1.2, yy= 1.1*y, h=.05*y)
        plot_significance(p_csp_csm_ctl, ax, x1=-.2, x2=.2, yy=1.1*y, h=.05*y)
        plot_significance(p_csm_csm, ax, x1=-.2, x2=.8, yy=1.25*y, h=.05*y)
        plot_significance(p_csp_csp, ax, x1=.2, x2=1.2, yy=1.4*y, h=.05*y)
    ax.set_ylabel(f'Lick/Test Session')
    ax.set_xlabel("")
    return ax, stats


def cumm_test_licks_plot(s, c, ax = None, mt_method = "holm-sidak", p_tot = None, palette ='magma', values = 'total_licks',
                         plot_sig = True, groups=[r'$AgRP^{Chr2}$', r'$AgRP^{TdTomato}$'], ms = 4, alpha = .3):
    """
    violin plot comparing total average testing day licks for stim and ctl
    
    Parameters:
    -----------
    s: pd.DataFrame
        the testing dataframe subset on the "stim" group
    c: pd.DataFrame
        the testing dataframe subset on the "control" group
    """
    if ax is None:
        _, ax = plt.subplots(1,1)
    if isinstance(palette, list):
        palette = palette[::-1]
    s, c = s.copy(), c.copy()
    s = s[values].unstack('CS').groupby('mouse').sum().sum(axis=1).rename(values).to_frame()
    c = c[values].unstack('CS').groupby('mouse').sum().sum(axis=1).rename(values).to_frame()
    df = pd.concat({groups[0]: s, groups[1]: c}, names=['Condition'])
    df = df.groupby(['Condition','mouse']).sum()
    sns.swarmplot(data = df.reset_index(), x='Condition', y=values, 
                  ax=ax, order = groups[::-1], color='k',alpha=alpha, size = ms)
    sns.violinplot(data = df.reset_index(), x ='Condition', y=values, linewidth = 0,
                    saturation = 10, order = groups[::-1],  ax=ax, palette =palette, alpha=.3 )
    ax.set_xlim(-.5,1.5)
    if p_tot is None:
        stat, p_tot = st.ttest_ind(df.loc[groups[0],values], 
                                   df.loc[groups[1],values],
                                   equal_var=False)
        deg_f = df.loc[groups[0],values].size + df.loc[groups[1],values].size - 2

    ax.set_ylabel('Cummulative Testing Licks')
    ax.set_xlabel('')
    sns.despine()

    if plot_sig:
        #plot the stats
        plot_significance(p_tot, ax, x1=0, x2=1,h=.1*df[values].max(),
                          yy=1.08*df[values].max())
    return ax, (stat, p_tot, deg_f)


def two_bottle_pref_plot_vl(s, c, ax = None, mt_method = "holm-sidak", p_pref = None, palette ='magma', values = 'total_licks', use_glm = False, 
                            ms = 4, alpha = .3, plot_sig = True, groups=[r'$AgRP^{Chr2}$', r'$AgRP^{TdTomato}$'], t=30):
    """
    violin plot comparing % cs+ licks stim v ctl
    
    Parameters:
    -----------
    s: pd.DataFrame
        the testing dataframe subset on the "stim" group
    c: pd.DataFrame
        the testing dataframe subset on the "control" group
    """
    if ax is None:
        _, ax = plt.subplots(1,1)
    if isinstance(palette, list):
        palette = palette[::-1]
    s, c = s.copy(), c.copy()
    s = s[values].unstack('CS').groupby('mouse').sum()
    c = c[values].unstack('CS').groupby('mouse').sum()
    df = pd.concat({groups[0]:s, groups[1]:c}, names = ['Condition'])
    df['tot'] = df['+']+df['-']
    df['pref'] = df['+']/(df['tot'])
    sns.swarmplot(ax=ax,data=df.reset_index(),  x='Condition', y='pref', 
                  order = groups[::-1], color='k',alpha=alpha, size = ms)
    sns.violinplot(ax=ax,data=df.reset_index(),  x='Condition',  y='pref', linewidth = 0, 
                    order = groups[::-1], saturation=10, palette =palette, alpha=.3, cut = 0)
    ax.set_xlim(-.5,1.5)

    ax.set_ylabel(f'Preference Index')
    ax.set_xlabel("")
    ax.set_ylim([-.1,1.4])
    sns.despine()

    if p_pref is None:
        if use_glm:
            _df = df.reset_index()
            model = smf.glm(f"pref ~ 1 + C(Condition, Treatment(reference=r'{groups[1]}'))", data = _df, 
                            family = sm.families.Binomial(), 
                            freq_weights = _df.tot).fit()
            p_pref = model.pvalues.loc[model.pvalues.index.str.contains("Condition")].iloc[0]
            res = model.summary()
        else:
            res = st.mannwhitneyu(df.loc[groups[0]]['pref'],df.loc[groups[1]]['pref'])
            _, p_pref = res
    if plot_sig:
        #plot the stats
        plot_significance(p_pref,ax,x1=0,x2=1,h=.10,yy=1.1)
    return ax, df, res



def training_plot(s,c, ax=None, groups=[r'$AgRP^{Chr2}$', r'$AgRP^{TdTomato}$'], colors = sns.color_palette('Set2', 2), ms=2, capsize=2, lw=1):
    """
    plot mean total licks over training
    
    Parameters:
    -----------
    s: pd.DataFrame
        the training dataframe subset on the "stim" group
    c: pd.DataFrame
        the training dataframe subset on the "control" group
    """
       
    if ax is None: _,ax=plt.subplots(1,1)
    mns_s = s.groupby(['CS','day']).total_licks.mean().sort_index()
    sems_s = s.groupby(['CS','day']).total_licks.sem().sort_index()

    mns_c = c.groupby(['CS','day']).total_licks.mean().sort_index()
    sems_c = c.groupby(['CS','day']).total_licks.sem().sort_index()
    
    x = list(mns_s.index.get_level_values('day').unique() + 1)
    
    # stim plots
    ax.errorbar(x, mns_s.loc['-'].values, sems_s.loc['-'].values, ls = '--',
                marker='o', capsize=capsize, lw=lw, ms = ms, color = colors[0],
                label = fr"{groups[0]} CS -")
    ax.errorbar(x, mns_s.loc['+'].values, sems_s.loc['+'].values, ls = '-',
                marker='o', capsize=capsize, lw=lw, ms = ms, color = colors[0],
                label = fr"{groups[0]} CS +")
    # ctl plots
    ax.errorbar(x, mns_c.loc['-'].values, sems_c.loc['-'].values, ls = '--',
                marker='o', capsize=capsize, lw=lw, ms = ms, color = colors[1],
                label = fr"{groups[1]} CS -")
    ax.errorbar(x, mns_c.loc['+'].values, sems_c.loc['+'].values, ls = '-',
                marker='o', capsize=capsize, lw=lw, ms = ms, color = colors[1],
                label = fr"{groups[1]} CS +")
    
    sns.despine(ax=ax)
    ax.set(xticks = x, xlabel = 'Training Day', ylabel ='Licks/Training Session');
    ax.legend(loc='lower left', bbox_to_anchor = (0,1), frameon = False, ncol = 2)
    return ax


def plot_t_to_end(s, c, ax = None, groups=[r'$AgRP^{Chr2}$', r'$AgRP^{TdTomato}$'], 
                  colors = sns.color_palette('Set1', 2)):
    """
    plot the time from infusion start to session end for each session
    
    Parameters:
    -----------
    s: pd.DataFrame
        the training dataframe of lick times subset on the "stim" group
    c: pd.DataFrame
        the training dataframe of lick times subset on the "control" group
    """
    
    if ax is None: _,ax=plt.subplots(1,1)
    def t_to_end(x):
        x = x.set_index('lick_number').sort_index()
        return x.time.iloc[-1] - x.time.loc[19]
    
    s = s.reset_index().groupby(['mouse','day','CS']).apply(t_to_end).rename('t')
    c = c.reset_index().groupby(['mouse','day','CS']).apply(t_to_end).rename('t')
    x = list(s.index.get_level_values('day').unique() + 1)
    
    # stim plots
    ax.errorbar(x, s.loc[:,:,'+'].groupby('day').mean(),
                s.loc[:,:,'+'].groupby('day').sem(), 
                marker = 'o', color = colors[0],
                capsize=4, lw=3, ms = 9, label = fr"{groups[0]} CS +")
    ax.errorbar(x, s.loc[:,:,'-'].groupby('day').mean(),
                s.loc[:,:,'-'].groupby('day').sem(), 
                marker = 'o', color = colors[0], ls = '--',
                capsize=4, lw=3, ms = 9, label = fr"{groups[0]} CS -")
    # ctl plots
    ax.errorbar(x, c.loc[:,:,'+'].groupby('day').mean(),
                c.loc[:,:,'+'].groupby('day').sem(), 
                marker = 'o', color = colors[1], 
                capsize=4, lw=3, ms = 9, label = fr"{groups[1]} CS +")
    ax.errorbar(x, c.loc[:,:,'-'].groupby('day').mean(),
                c.loc[:,:,'-'].groupby('day').sem(), 
                marker = 'o', color = colors[1], ls = '--',
                capsize=4, lw=3, ms = 9, label = fr"{groups[1]} CS -")
    sns.despine(ax=ax)
    ax.set(xticks = x, xlabel = 'Training Day', ylabel ='Time From Infusion End to Session End (s)');
    ax.legend(loc='lower left', bbox_to_anchor = (0,1), frameon = False, ncol = 2)
    
    return ax, s, c
    

def plot_sex_diff_pref(pref, ax = None, mt_method = "holm-sidak", palette ='magma', 
                      groups=[r'$AgRP^{Chr2}$', r'$AgRP^{TdTomato}$']):
    
    #plot preference by sex
    pref = pref.copy()
    pref['x'] = (0.5*pref.Chr2.astype(int)) + (2*(pref.sex=='F').astype(int) - 1) + 0.05*np.random.randn(len(pref))
    pref['Total Lick Bursts'] = pref['tot']
    pref['Strain'] = pref['Chr2'].apply(lambda x: groups[::-1][x])

    if ax is None:
        _, ax = plt.subplots(1,1)
    g = sns.scatterplot(pref, x='x', y='pref', size = 'Total Lick Bursts', hue = 'Strain', palette = palette, ax=ax)
    xmn = pref.groupby(['Chr2','sex']).x.mean()
    pmn = pref.groupby(['Chr2','sex']).apply(lambda x: (x.tot@x.pref)/x.tot.sum())
    for i,j in zip(xmn, pmn):
        g.plot([i-0.1, i+0.1], [j,j], color = 'k', alpha=0.5)


    g.set_xlim(-1.8,2.3)
    g.set_xticks([-0.75, 1.25])
    g.set_xticklabels(['Male', 'Female'])
    g.set_xlabel("")
    g.set_ylabel("Burst Preference Index")
    g.legend(loc = 'upper left', bbox_to_anchor = (1,1), frameon=False)



    # compute stats
    sex_comp = (pref.groupby('Chr2')
                    .apply(lambda x: smf.glm('pref ~ 1 + sex', data = x.reset_index(), 
                                             family = sm.families.Binomial(), 
                                             freq_weights = x.tot).fit())
                    .apply(lambda x:  pd.Series({'coef': x.params['sex[T.M]'].astype(str) + "±" + x.bse['sex[T.M]'].astype(str) , 
                                                 't': x.tvalues['sex[T.M]'],
                                                 'CI': x.conf_int().apply(lambda x: x.round(4).astype(str).loc[0] + ', ' + x.round(4).astype(str).loc[1], axis=1).loc['sex[T.M]'],
                                                 'p-value': x.pvalues['sex[T.M]'],
                                                 'df': x.df_resid}))
                    .reset_index()
    )

    stim_comp = (pref.groupby('sex')
                     .apply(lambda x: smf.glm('pref ~ 1 + Chr2', data = x, 
                                              family = sm.families.Binomial(), 
                                              freq_weights = x.tot).fit())
                     .apply(lambda x:  pd.Series({'coef': x.params['Chr2[T.True]'].rounastype(str) + "±" + x.bse['Chr2[T.True]'].astype(str), 
                                                  't': x.tvalues['Chr2[T.True]'], 
                                                  'CI': x.conf_int().apply(lambda x: x.round(4).astype(str).loc[0] + ', ' + x.round(4).astype(str).loc[1], axis=1).loc['Chr2[T.True]'],
                                                  'p-value': x.pvalues['Chr2[T.True]'],
                                                  'df': x.df_resid}))
                     .reset_index()
    )

    pref_sex_diff_stats = pd.concat({'Male vs Female': sex_comp, 
                                         'Stim vs Control': stim_comp}, 
                                        names = ['comparison']).droplevel(1).reset_index()
    pref_sex_diff_stats['Corrected p-value'] = multipletests(pref_sex_diff_stats['p-value'], method = mt_method)[1]
    pcorr = pref_sex_diff_stats.set_index(['comparison', 'Chr2', 'sex'])['Corrected p-value']

    # plot the stats
    y=pref.pref.max()
    plot_significance(pcorr.droplevel('sex').loc['Male vs Female', True].iloc[0],
                      g, x1=-0.5, x2=1.5, yy=1.4*y, h=.05*y)
    plot_significance(pcorr.droplevel('sex').loc['Male vs Female', False].iloc[0],
                      g, x1=-1, x2=1, yy=1.25*y, h=.05*y)
    plot_significance(pcorr.droplevel('Chr2').loc['Stim vs Control', 'F'].iloc[0], 
                      g, x1=1, x2=1.5, yy= 1.1*y, h=.05*y)
    plot_significance(pcorr.droplevel('Chr2').loc['Stim vs Control', 'M'].iloc[0], 
                      g, x1=-1, x2=-.5, yy=1.1*y, h=.05*y)

    sns.despine(ax=g)
    
    return g, pref_sex_diff_stats    

def plot_sex_diff(df, key, ax = None, mt_method = "holm-sidak", palette ='magma', 
                  groups=[r'$AgRP^{Chr2}$', r'$AgRP^{TdTomato}$'], ylabel = None):
    
    if ax is None:
        _, ax = plt.subplots(1,1)
    if ylabel is None:
        ylabel = key
    # glucose
    # plot training licks split by strain and sex
    sns.swarmplot(data = df, x = 'sex', y = key, hue = 'Chr2', 
                  order = ['M','F'], hue_order=[False,True],
                  dodge = True, palette = 'dark:k', alpha = .5,  ax = ax, legend = False)
    sns.violinplot(data =df, x = 'sex', y = key, hue = 'Chr2', 
                   saturation = .9, order = ['M','F'], hue_order=[False,True],
                   linewidth = 0, palette = palette, ax = ax)
    handles, labels = ax.get_legend_handles_labels() 
    ax.legend(handles, groups[::-1], loc = 'lower left', 
             frameon = False, bbox_to_anchor=(0,1.1))
    ax.set_xticklabels(["Male", "Female"], fontsize = 8)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel, fontsize = 8)


    # compute stats
    sex_comp = (df.set_index(['Chr2','sex','mouse']).groupby('Chr2')[key]
                  .apply(lambda x: st.ttest_ind(x.loc[:,'M'], x.loc[:,'F'], equal_var = False) + (x.shape[0] - 2,))
                  .apply(lambda x: pd.Series(x, index = ('t', 'p-value', 'df')))
               )
    stim_comp = (df.set_index(['Chr2','sex','mouse']).groupby('sex')[key]
                   .apply(lambda x: st.ttest_ind(x.loc[True,], x.loc[False,], equal_var = False) + (x.shape[0] - 2,))
                   .apply(lambda x: pd.Series(x, index = ('t', 'p-value', 'df')))
                )

    sex_diff_stats = pd.concat({'Male vs Female': sex_comp.reset_index(), 
                                'Stim vs Control': stim_comp.reset_index()},
                               names = ['comparison']).droplevel(1).reset_index()
    sex_diff_stats['Corrected p-value'] = multipletests(sex_diff_stats['p-value'], method = mt_method)[1]
    pcorr = sex_diff_stats.set_index(['comparison', 'Chr2', 'sex'])['Corrected p-value']

    # plot the stats
    y=df[key].max()
    plot_significance(pcorr.droplevel('sex').loc['Male vs Female', True].iloc[0],
                      ax, x1=.2, x2=1.2, yy=1.4*y, h=.05*y)
    plot_significance(pcorr.droplevel('sex').loc['Male vs Female', False].iloc[0],
                      ax, x1=-.2, x2=.8, yy=1.25*y, h=.05*y)
    plot_significance(pcorr.droplevel('Chr2').loc['Stim vs Control', 'F'].iloc[0], 
                      ax, x1=.8, x2=1.2, yy= 1.1*y, h=.05*y)
    plot_significance(pcorr.droplevel('Chr2').loc['Stim vs Control', 'M'].iloc[0], 
                      ax, x1=-.2, x2=.2, yy=1.1*y, h=.05*y)
    # ax.set_xlim(-.5,1.5)
    return ax, sex_diff_stats


def plot_training_micro(burst_summ, stat, palette = 'Set1', ylabel = None, ax = None):
    ylabel = stat if ylabel is None else ylabel
    mean = burst_summ.groupby(['Chr2', 'CS', 'day']).mean()
    sem = burst_summ.groupby(['Chr2', 'CS', 'day']).sem()

    pal = sns.color_palette(palette, 2)
    
    if ax is None:
        _,ax = plt.subplots(1,1)

    ax.errorbar(mean.loc[False, '+'].reset_index().day, mean.loc[False, '+'][stat],
                 sem.loc[False, '+'][stat], marker = 'o', capsize = 3, 
                 c = pal[1], label = 'Control CS+')
    ax.errorbar(mean.loc[False, '-'].reset_index().day, mean.loc[False, '-'][stat],
                 sem.loc[False, '-'][stat], marker = 'o', capsize = 3, 
                 c = pal[1], ls = '--', label = 'Control CS-')

    ax.errorbar(mean.loc[True, '+'].reset_index().day, mean.loc[True, '+'][stat],
                sem.loc[True, '+'][stat], marker = 'o', capsize = 3, 
                c = pal[0], label = 'Stim CS+')
    ax.errorbar(mean.loc[True, '-'].reset_index().day, mean.loc[True, '-'][stat],
                sem.loc[True, '-'][stat], marker = 'o', capsize = 3, 
                c = pal[0], ls = '--', label = 'Stim CS-')

    sns.despine()
    ax.set(xticks = mean.reset_index().day.unique(),
           xticklabels =  mean.reset_index().day.unique().astype(int) + 1,
           xlabel = 'Training Day',
           ylabel = ylabel);
    ax.legend(loc = 'lower left', bbox_to_anchor = (0,1), ncol = 2, frameon = False)
    
    return ax

############################
## FUNCTIONS FOR STATS ##
############################

def model_to_str(model, dec = 4):
    beta = 'β=' + model.params.round(dec).astype(str)
    ci =  model.conf_int().round(dec).astype(str)
    ci = '95% CI [' + ci.iloc[:,0] + ', ' + ci.iloc[:,1] + ']'
    if model.use_t:
        stat = f't({model.df_resid})=' + model.tvalues.round(dec).astype(str)
    else:
        stat = 'z=' + (model.params/model.bse).round(dec).astype(str)
    pvals = model.pvalues
    def eval_p(p):
        if p<.0001:
            return "p<.0001"
        elif p<.001:
            return "p<.001"
        elif p<.01:
            return "p<.01"
        else:
            return f"p={p:.2f}"
    pstr = pvals.apply(eval_p)
    res = (beta + '; ' + ci + '; ' + stat + '; ' + pstr).rename('Result').to_frame()
    res['Significance'] = pvals.apply(check_significance)
    return res

def check_significance(p):
    """
    get the appropriate number of asterisks to plot
    given the pvalue
    """
    crit_vals=np.array([.05,.01,.001,.0001])
    sig_labels=['ns','*','**','***','****']
    return sig_labels[np.digitize(p,crit_vals)]

def plot_significance(p, ax, x1, x2, yy, h):
    """
    plot the significance
    """
    x = [x1, x1, x2, x2]
    y = [yy, yy+h, yy+h, yy]
    x_text, y_text = (x1+x2)/2, yy+h
    ax.plot(x, y,color='k', linewidth=.5)
    ax.text(x_text, y_text, check_significance(p), 
            ha='center', va='bottom')


def two_bottle_stats(s, c, mt_method = "holm-sidak", all_paired = False, groups = ['stim', 'ctl']):
    """
    compute stats for average two bottle tests results
    """
    s = s.unstack("CS").copy()
    c = c.unstack("CS").copy()
    
    #compare cs+/cs- licks
    r = st.ttest_rel(s['+'], s['-'])
    t_csp_csm_stim, p_csp_csm_stim = r
    df_csp_csm_stim = r.df
    
    r = st.ttest_rel(c['+'], c['-'])
    t_csp_csm_ctl, p_csp_csm_ctl = r
    df_csp_csm_ctl = r.df
    
    r = st.ttest_ind(s['+'], c['+'], equal_var=False) if not all_paired else st.ttest_rel(s['+'], c['+'])
    t_csp_csp, p_csp_csp = r
    df_csp_csp = (s['+'].size + c['+'].size) - 2 if not all_paired else r.df
    
    r = st.ttest_ind(s['-'], c['-'], equal_var=False) if not all_paired else st.ttest_rel(s['+'], c['+'])
    t_csm_csm, p_csm_csm = r
    df_csm_csm = (s['-'].size + c['-'].size) - 2 if not all_paired else r.df
    
    
    #multiple comparisons test
    _, (p_csp_csm_stim_corr, p_csp_csm_ctl_corr,
        p_csp_csp_corr, p_csm_csm_corr), _, _ = multipletests([p_csp_csm_stim,
                                                               p_csp_csm_ctl, 
                                                               p_csp_csp, 
                                                               p_csm_csm], method = mt_method)
    
    res = pd.DataFrame({"statistic":   [t_csp_csm_stim, t_csp_csm_ctl, t_csp_csp, t_csm_csm],
                        "pvalue"   :   [p_csp_csm_stim, p_csp_csm_ctl, p_csp_csp, p_csm_csm],
                        "pvalue_corr": [p_csp_csm_stim_corr, p_csp_csm_ctl_corr, p_csp_csp_corr, p_csm_csm_corr],
                        "df" :         [df_csp_csm_stim, df_csp_csm_ctl, df_csp_csp, df_csm_csm],
                        "paired": [True, True, False, False] if not all_paired else [True, True, True, True]},
                        index = [f'CS+ vs. CS- {groups[0]}', 
                                 f'CS+ vs. CS- {groups[1]}', 
                                 f'CS+ {groups[0]} vs. CS+ {groups[1]}', 
                                 f'CS- {groups[0]} vs. CS- {groups[1]}'])

    return res

