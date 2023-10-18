import sys
photometry_scripts_path = "/Users/nathanielnyema/Downloads/photometry-scripts"
sys.path.append(photometry_scripts_path)
from analysis_pipeline import *
import matplotlib as mpl
from scipy import signal, ndimage, optimize
from matplotlib import pyplot as plt
import statsmodels.formula.api as smf
from tqdm import tqdm
from scipy.special import expit
from scipy import optimize


# experiment parameters
cs = {'+': ['training_1_pm','training_2_am','training_3_pm'],
      '-': ['training_1_am','training_2_pm','training_3_am']}

mp = np.array(['-','+'])


# useful functions for peri-bout analyses

def peri_event(a, event, w, df = None, norm = 'none', base_len = None):
    """
    
    """   
    
    if df is None:
        df = a.all_490.copy()
    if base_len is None:
        base_len = w
    b = a.raw_data.droplevel('trial')
    x = np.arange(w + 1/a.ds_freq, step = np.round(1/a.ds_freq,2),  dtype=float)
    x = np.concatenate((-x[1:][::-1], x))
    x = np.round(pd.Index(x),3)
    end_lim = df.index[-1]-w-1
    cols = []
    for i,v in b.items():
        cols.extend([i+ (j,) for j in np.arange((v.events[event] < end_lim).sum())])
    cols = pd.MultiIndex.from_tuples(cols,names = ['cond','mouse','event'])
    d = pd.DataFrame(np.ones((x.size,cols.size))*np.nan, 
                     columns = cols, 
                     index = pd.Series(x))
    for i,v in b.items():
        for j,v2 in enumerate(v.events[event]):
            if (v2 < end_lim):
                d.loc[:,pd.IndexSlice[i+(j,)]] = df.loc[v2-w:v2 + w + 1/a.ds_freq][i].values
                
    if norm =='sub_mean':
        d -= np.mean(d,axis=0) 
    elif norm == 'sub_med_prestim':
        assert base_len is not None, "must specify base_len arg"
        d -= np.median(d.loc[:d.index.min() + base_len], axis=0)
    elif norm == 'sub_center':
        d -= d.loc[0]
    elif norm == 'norm_median':
        d = (d- np.median(d, axis=0))/np.median(d, axis = 0)
    elif norm == 'zscore':
        d = (d- np.mean(d, axis=0))/np.std(d, axis = 0)
    elif norm == 'none':
        return d
    else:
        print("unrecognized sub")
        return
    return d

def reindex(p, c = None):
    p_n = p.T.reset_index()
    if c is None:
        cs_idx = pd.Series(['+'] * p_n.index.size)
        cs_idx.loc[p_n.cond.isin(cs['-'])] = '-'
    else:
        cs_idx = pd.Series([c] * p_n.index.size)
    if 'event' in p_n.columns:
        new_idx = list(zip(p_n.cond.str.extract(r'(\d)', expand = False).astype(int).values, 
                       cs_idx.values, p_n.mouse.values, p_n.event ))
        new_idx = pd.MultiIndex.from_tuples(new_idx, names = ['day','cs','mouse','event'])
    else:
        new_idx = list(zip(p_n.cond.str.extract(r'(\d)', expand = False).astype(int).values, 
                       cs_idx.values, p_n.mouse.values))
        new_idx = pd.MultiIndex.from_tuples(new_idx, names = ['day','cs','mouse'])
    p_n = p_n.set_index(new_idx).sort_index()
    p_n.columns.name = 'time'
    
    if 'event' in p_n.columns:
        return p_n.drop(['cond','mouse','event'], axis = 1)
    else:
        return p_n.drop(['cond','mouse'], axis = 1)

def randomize(a, event, ev_thresh, w, niters, norm = 'none', base_len = None, trange = (None, None), select_evs = None):
    """
    randomly sample times for bout starts such that no
    2 samples are less than some minimum value from each other
    """
    d = None
    end_lim = a.all_490.index[-1]-w-1
    end_lim = min(end_lim, trange[1]) if trange[1] else end_lim
    start_lim = max(0, trange[0]) if trange[0] else 0
    for k in tqdm(range(niters)):
        for i,v in a.raw_data.items():
            ts = np.arange(0, a.all_490.index[-1] + 1/100, step = 1/100)
            ts = ts[(ts>start_lim) & (ts < end_lim) ]
            n = v.events[event].size
            rand_evs = np.empty((n,))
            for j in range(n):
                rand_evs[j] = np.random.choice(ts)
                ts = ts[np.abs(ts - rand_evs[j]) > ev_thresh]
            rand_evs = np.sort(rand_evs)
            if select_evs:
                rand_evs = select_evs(rand_evs)
            a.raw_data.loc[i].events[f'rand_{event}'] = rand_evs
        _d = peri_event(a, f'rand_{event}',  w, norm = norm, base_len = base_len).groupby(['cond','mouse'], axis=1).mean()
        d = d +_d if d is not None else _d
    return d/niters


def plot_peri(mn, ctl = None, phase="Training", saveroot = "", savename = "", figsize = (7,5), save = False,
              vmin = -1, vmax = 1, ylim = (-1,1), linewidth = 1, vlabel = 'Z-Score', ylabel = r'Z-Scored $\frac{\Delta F}{F}$'):
    
    mn = mn.sort_index()
    w = mn.columns.max()
    days = mn.index.get_level_values('day').unique()

    fig, ax = plt.subplots(3,days.max(), figsize = figsize, gridspec_kw={'height_ratios': [.8, 1, 1]})
    cbar_ax = fig.add_axes([.95, .2, .03, .4])
    for i in days:
        ax[0,i-1].plot(mn.loc[i,'+'].mean(axis=0), label = 'CS+', lw=linewidth)
        ax[0,i-1].fill_between(mn.columns.astype(float), alpha = .3,
                         y1= (mn.loc[i,'+'].mean(axis=0) - mn.loc[i,'+'].sem(axis=0)), 
                         y2 = (mn.loc[i,'+'].mean(axis=0) + mn.loc[i,'+'].sem(axis=0)))
        ax[0,i-1].plot(mn.loc[i,'-'].mean(axis=0), label = f'CS-', lw=linewidth)
        ax[0,i-1].fill_between(mn.columns.astype(float), alpha = .3,
                         y1= (mn.loc[i,'-'].mean(axis=0) - mn.loc[i,'-'].sem(axis=0)), 
                         y2 = (mn.loc[i,'-'].mean(axis=0) + mn.loc[i,'-'].sem(axis=0)))
        if ctl is not None:
            ax[0,i-1].plot(ctl.loc[i,'+'].mean(axis=0), c='gray', label = 'CS+ randomized', lw=linewidth)
            ax[0,i-1].fill_between(ctl.columns.astype(float), alpha = .3, color = 'gray',
                             y1= (ctl.loc[i,'+'].mean(axis=0) - ctl.loc[i,'+'].sem(axis=0)), 
                             y2 = (ctl.loc[i,'+'].mean(axis=0) + ctl.loc[i,'+'].sem(axis=0)))
            ax[0,i-1].plot(ctl.loc[i,'-'].mean(axis=0), c='lightgray', label = 'CS- randomized', lw=linewidth)
            ax[0,i-1].fill_between(ctl.columns.astype(float), alpha = .3, color = 'lightgray',
                             y1= (ctl.loc[i,'-'].mean(axis=0) - ctl.loc[i,'-'].sem(axis=0)), 
                             y2 = (ctl.loc[i,'-'].mean(axis=0) + ctl.loc[i,'-'].sem(axis=0)))
        ax[0,i-1].set_title(f'{phase} Day {i}')
        ax[0,i-1].set_xlabel('');
        ax[0,i-1].set_ylim(ylim);
        ax[0,i-1].set_xlim(-w,w);
        ax[0,i-1].set_xticks([]);
        
        ax[0,i-1].axvline(0, c='gray', ls='--')
        
        sns.heatmap(mn.loc[i,'-'].sort_index(), ax = ax[1,i-1], vmin = vmin, vmax = vmax, 
                    cbar = None if i<days.max() else True, cbar_ax = cbar_ax,cbar_kws={'label': vlabel} )
        ax[1,i-1].axvline(np.where(mn.columns==0)[0], c = 'k', ls = '--')
        ax[1,i-1].set_xticks([])
        ax[1,i-1].set_xlabel('')
        ax[1,i-1].set_ylabel('')
        
        sns.heatmap(mn.loc[i,'+'].sort_index(), ax = ax[2,i-1], vmin = vmin, vmax = vmax, cbar = None)
        ax[2,i-1].axvline(np.where(mn.columns==0)[0], c = 'k', ls = '--')
        ax[2,i-1].set_ylabel('')
        ax[2,i-1].set_xlabel('Time Lag (s)')
        ax[2,i-1].set_xticks(np.where(mn.columns.isin([-w,-w//2,0,w//2,w]))[0])
        ax[2,i-1].set_xticklabels(np.array([-w,-w//2,0,w//2,w]).astype(int))
        
        if i>1: 
            ax[0,i-1].set_yticks([])
        ax[1,i-1].set_yticks([])
        ax[2,i-1].set_yticks([])
        
    ax[1,0].set_ylabel('CS-')
    ax[2,0].set_ylabel('CS+')
    ax[0,0].set_ylabel(ylabel);
    ax[0,0].set_ylim(ylim);
    # ax[0,-1].legend(loc = 'upper left', bbox_to_anchor = (1,1), frameon = False)
    # fig.tight_layout(pad = .4)
    
    if save:
        fig.savefig(os.path.join(saveroot, 'pdfs', savename + '.pdf'), bbox_inches = 'tight')
        fig.savefig(os.path.join(saveroot, 'svgs', savename + '.svg'), bbox_inches = 'tight')

    return fig, ax

def time_lock_lick(an, lick_idx = 0, lick_field = 'all_licks'):
    """
    function for time locking each traces to a specified lick during the corresponding session
    """
    a = deepcopy(an)
    max_shift = round(max([i.events[lick_field][lick_idx] for i in a.normed_data])) * a.ds_freq
    tmp = pd.DataFrame([], columns = a.all_490.columns)
    tmp2 = pd.DataFrame([], columns = a.all_405.columns)

    for i in a.normed_data:

        x = a.all_490[pd.IndexSlice[i.cond,i.mouse_id]].values
        x = np.roll(x, -round(i.events[lick_field][lick_idx])* a.ds_freq )[:-max_shift]
        tmp[pd.IndexSlice[i.cond,i.mouse_id]] = pd.Series(x, index = np.arange(-a.t_prestim, 1800 - (max_shift-1)/a.ds_freq, step = 1/a.ds_freq))

        y = a.all_405[pd.IndexSlice[i.cond,i.mouse_id]].values
        y = np.roll(y, -round(i.events[lick_field][lick_idx])* a.ds_freq )[:-max_shift]
        tmp2[pd.IndexSlice[i.cond,i.mouse_id]] = pd.Series(y, index = np.arange(-a.t_prestim, 1800 - (max_shift-1)/a.ds_freq, step = 1/a.ds_freq))

    # #update the dataframes manually
    a.all_490 = tmp.copy()
    a.mean_490 = tmp.groupby('cond', axis = 1).mean(numeric_only=True)
    a.err_490 = tmp.groupby('cond', axis = 1).sem(numeric_only=True)
    a.t_endrec = tmp.index[-1]

    a.all_405 = tmp2.copy()
    a.mean_405 = tmp2.groupby('cond', axis = 1).mean(numeric_only=True)
    a.err_405 = tmp2.groupby('cond', axis = 1).sem(numeric_only=True)

    return a


def get_lick_rate (a, event, sigma = .1, win = .5):
    ilis = []
    for i in a.raw_data:
        ll = i.events[event]
        ilis.append((ll[1:] - ll[:-1]).min())
    fs = np.ceil(2/min(ilis))
    t = np.arange(-a.t_prestim, a.t_endrec + 1/fs, step = 1/fs)
    hist = a.raw_data.map(lambda x: np.append(np.histogram(x.events[event], t)[0], 0))
    lr = pd.DataFrame(np.array([i for i in hist.values]).T, index = t, columns = hist.index)
    t_k = np.arange(-win, win + (1/fs), step = 1/fs)
    kernel = np.exp(-0.5 * (t_k/sigma)**2)
    kernel = fs * kernel/kernel.sum()
    lr = lr.apply(lambda x: pd.Series(np.convolve(  x, kernel, mode = 'same'), index = t), axis = 0)
    return lr.reindex(a.t, method = 'nearest').droplevel('trial', axis=1)
        
def bin_peri_bout(df, binsize = 2):
    melted = df.stack().rename('df').reset_index()
    w = melted.time.max()
    bins = np.arange(w + 0.0001,  step =binsize)
    bins = np.concatenate((-bins[-1:0:-1], bins))
    melted.loc[(melted.time<bins[0]) |(melted.time>bins[-1]), 'time']  = np.nan
    d = np.digitize(melted.time, bins, right=False) - 1
    invalid = d==bins.size
    d[invalid] = 0
    t = bins[d]
    t[invalid] = np.nan
    melted['bins'] = t
    if 'event' in melted.columns:
        return melted.groupby(['day','cs','mouse','event','bins']).df.mean().unstack('bins')
    else:
        return melted.groupby(['day','cs','mouse','bins']).df.mean().unstack('bins')

def get_bn(x, lick_event, bout_event):
    licks = x.events[lick_event]
    bouts = x.events[bout_event]
    if len(bouts)>0:
        licks = licks[licks>= bouts[0]] # should only consider licks starting after the first of the given bout
        # this is especially important for testing where a bout may contain licks from
        bout_label = np.digitize(licks, bouts, right=False)-1
        bns = np.array([(bout_label==i).sum() for i in np.unique(bout_label)])
        return bns
    else:
        return np.array([])
        

def agrp_model(x, h1, w1, loc1, ret, w2, dloc2):
    b1 = np.log((2*np.sqrt(6) + 5))/(w1/2)
    sig1 =  - h1 * expit(b1 * (x - loc1))
    sig1_bot = loc1 + w1/2
    b2 = np.log((2*np.sqrt(6) + 5))/(w2/2)
    h2 = ret * h1
    loc2 = sig1_bot+ w2/2 + dloc2
    sig2 = h2 * expit(b2*(x - loc2))
    return sig1 + sig2

def get_params(x, y):
    bounds = ([-10, 300, 0, 0, 300, 0], [10, 1800, 1800, 2, 1800, 1800])
    return optimize.curve_fit(agrp_model, x, y, bounds = bounds)

def fit_model(y, x=None, normed = False):
    if x is None:
        assert isinstance(y, pd.Series), "if x is not specified y must be a pandas series"
        x = y.index
        y = y.values
    if not normed: 
        f0 = np.median(y[x<0])
        popt, pcov  = get_params(x, (y - f0)/f0)
        return  agrp_model(x, *popt)*f0 + f0
    else:
        popt, pcov  = get_params(x, y)
        return  agrp_model(x, *popt) 