import numpy as np
from numpy import matlib as mat
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import datetime
from pathlib import Path 
import re
from scipy import stats as st
import pandas as pd
from pull_operant_data import pull_data



class FNCData():

    """
    a data loading class to load data from a given phase of our lab's protocol
    for flavor-nutrient learning (habituation, training, pre-testing, testing).
    
    data is assumed to be organized such that the folder specified by fpath
    only contains data from this phase of the experiment. sub-folders should be named
    after the date the data was collected, and within these subfolders should be the 
    folders of data for each session of that day. Note, that for days where there were 2 sessions
    we assume that there is an 'am' session and a 'pm' session and search for these indicators 
    in the folder name to distinguish between the two.

    another point worth mentioning is it is assumed here that a mouse stays in the same box throughout an experiment
    its easy to forget to enter the subject name in med-pc so we use the box assignments to figure out who's who. this
    is also important for behavior anyways so generally try to keep mice in the same box!

    Attributes
    ----------


    """

    def __init__(self, fpath:Path, name_prefix:list, phase:str, setup_file, 
                n_sessions=1, loglevel=1):
        """
        Inputs
        ------
        fpath: pathlib.Path, str
            path to the folder containing all of this experiment's data
        name_prefix: list
            a list of possible prefixes for mouse names (for us usually the initials of the surgeon)
        phase: str
            the name of the folder containing the data for the phase of the experiment you'd like to load
        name_box: Dict
            a dictionary mapping each mouse to the box it was in for the experiment. names are keys, 
            boxes are the entries
        
        Outputs
        -------
        self
        """

        fpath = Path(fpath).resolve()
        ps = [ p for p in (fpath/phase).resolve().iterdir() if p.is_dir() ]
        ps.sort()
        day_folders = [[ (p.parent/i).resolve() for i in p.iterdir() if i.name[0]!='.'] for p in ps]
        setup = pd.read_csv(setup_file, index_col = 0)
        name_box = setup.box.to_dict()

        #initialize dictionaries to store the lick data
        if n_sessions == 1:
            total_right_licks = { k: [None] * len(ps) for k in name_box.keys() }
            total_left_licks = { k: [None] * len(ps) for k in name_box.keys() }
            right_licks = { k: [None] * len(ps) for k in name_box.keys() }
            left_licks = { k: [None] * len(ps) for k in name_box.keys() }

        # if this was a phase of the experiment where we ran 2 sessions 
        elif n_sessions==2:
            total_right_licks = { k : [None] * len(ps) * 2 for k in name_box.keys() }
            total_left_licks = { k : [None] * len(ps) * 2 for k in name_box.keys() }
            right_licks = { k : [None] * len(ps) * 2 for k in name_box.keys() }
            left_licks = { k : [None] * len(ps) * 2 for k in name_box.keys() }


        for day, day_folder in enumerate(day_folders): #iterate over the folders for each day
            for folder in day_folder: #iterate over all folders in a given day
                #pull out mice ids for this folder
                mice = re.findall('\d+.|'.join(name_prefix)+'\d+.',folder.name.lower()) 
                if loglevel==2: print(folder.name)
                mice = list(map( lambda m:  m if m[-1] in 'ab' else m[:-1], mice))
                 #identify which box each mouse will be in
                box_name = {}
                for m in mice:
                    if m not in name_box: 
                        if loglevel>1:
                            print(f'WARNING: mouse {m} was not found in the setup file. {folder.as_posix()}')
                    else:
                        box_name.update({name_box[m]: m})
                for f in folder.iterdir():
                    # iterate over all files saved for this session
                    if f.is_file() and f.name[0]!='.': # make sure its actually a file
                        try:
                            #pull the data
                            data = pull_data(f)
                            if len(data)>1:
                                if  loglevel>0: 
                                    print(f"WARNING: {len(data)} boxes found in this file {f.as_posix()}")
                            for box, d in data.items():
                                if len(d)>1: 
                                    if  loglevel>0:
                                        print( f"WARNING: {len(d)} recordings found in file {f.as_posix()}", 
                                               "; end date of most recent:", data[box][-1]['end'],)
                                if box in box_name:
                                    d=d[-1] # select the most recent data for this box
                                    name = box_name[box] # figure out which mouse it was based on the box
                                    if (d['subject']!=name) and (loglevel>0):
                                        print( f"WARNING: {d['subject']} was found as subject but {name} was expected for box {box}; {f.as_posix()}")
                                    pm = 1 if 'pm' in folder.name.lower() else 0
                                    sess = n_sessions * day + pm
                                    if loglevel>2: print(sess)
                                    total_right_licks[name][sess] = d['b'][0, 0]
                                    total_left_licks[name][sess] = d['a'][0, 0]
                                    right_licks[name][sess] = d['r'].flatten()
                                    left_licks[name][sess] = d['l'].flatten()
                                elif loglevel>1:
                                    print(f"WARNING: data found for box {box} but no mice in the folder name",
                                          f"were expected to be in this box from the setup file; {f.as_posix()}")
                        except Exception as e:
                            raise Exception(f"something went wrong with this file {f.as_posix()} - {e}")
        self.phase = phase
        self.name_box = name_box
        self.names = list(name_box.keys())
        self.total_right_licks = (pd.DataFrame(total_right_licks).T
                                    .rename_axis(index =  "mouse", columns = "cumm_sess"))
        self.total_left_licks  = (pd.DataFrame(total_left_licks).T
                                   .rename_axis(index =  "mouse", columns = "cumm_sess"))
        self.total_licks = self.total_right_licks + self.total_left_licks
        self.right_licks = (pd.DataFrame(right_licks).T
                              .rename_axis(index =  "mouse", columns = "cumm_sess"))
        self.left_licks  = (pd.DataFrame(left_licks).T
                              .rename_axis(index =  "mouse", columns = "cumm_sess"))

        if phase=='pretesting':
            self.pside = setup[['pretest_am_side','pretest_pm_side']].copy()
            self.mside = setup[['pretest_am_side','pretest_pm_side']].copy()
            self.pside.columns = [0,1]
            self.mside.columns = [1,0]
            self.pside.columns.name = 'sess'
            self.mside.columns.name = 'sess'
        elif phase == 'testing':
            ds = [setup.test_day_1_side, 
                  setup.test_day_1_side.replace({'right':'left', 'left':'right'})]
            self.pside =  pd.concat({i: ds[i%2] for i in range(self.total_left_licks.columns.size)}, axis=1).dropna()
            self.mside =  self.pside.replace({'right':'left', 'left':'right'})
            self.pside.columns.name = 'day'
            self.mside.columns.name = 'day'
        
        self.setup = setup
        if loglevel==2: print('done')


""" 
The following are functions for compiling the data from multiple FNCData objects 
into larger dataframes for further analyses. Some of the functions create a 
dataframe of total licks per session and create a dataframe
with information about lick times.

These dataframes are stored in long format. So for total licks, each row corresponds
to a total lick measurement, and other columns specify which mouse, CS, day within 
the given phase, and session within the given day this measurement corresponds to as 
well as whether or not the mouse had ChR2, the phase of the experiment, bottle side 
and what US we were using (glucose or fat) .

Similarly, for lick times, each row is a lick and we specify the time of the lick,
which lick in the session it is, and otherwise the same information as for total licks
"""



def get_total_testing_licks(a,chr2,us,df_i=None, pretest = False):
    """
    add all the total licks from a given FNCData object with testing    
    data to a large dataframe to store all the data from an experiment

    Inputs
    ------
    a: FNCData
    us: str
        what the unconditioned stimulus was for the CS+ flavor for this exp
        (i.e. glucose or fat)
    chr2: bool
        whether or not these were channelrhodopsin mice
    df_i: pandas.DataFrame
        the dataframe to add the data to. if not specified we'll create a new one
    
    Outputs
    -------
    df: pandas.DataFrame
        the compiled dataframe
    """

    
    df = (pd.concat({ 'left'  : a.total_left_licks.dropna(),
                      'right' : a.total_right_licks.dropna() },
                    names = ['side'])
            .rename_axis("day", axis=1)
            .stack("day")
            .reorder_levels(["mouse", "day", "side"])
            .rename("total_licks")
            .to_frame()
            .sort_index())
    
    if pretest:
        psides = a.pside.stack("sess").rename("side").to_frame().reset_index().sort_index()
    else:
        psides = a.pside.stack("day").rename("side").to_frame().reset_index().sort_index()
    for mouse, day, pside in psides.itertuples(index = False):
        mside = "left" if pside == "right" else "right"
        df.loc[(mouse, day, pside), "CS" ] = '+'
        df.loc[(mouse, day, mside), "CS" ] = '-'
        
    df = df.reset_index()
    df['sex']   = a.setup.loc[df.mouse,'sex'].values
    df['CS+']   = a.setup.loc[df.mouse,'CS+'].values
    df['CS-']   = a.setup.loc[df.mouse,'CS-'].values
    df['sess']  = 0
    df['phase'] = 'test'
    df['US']    = us
    df['Chr2']  = chr2
    
    df = df.set_index(['US','Chr2','mouse','phase','day','CS'])
    
    if df_i is None:            
        return df
    else:
        return pd.concat([df_i, df])


def get_total_training_licks(a,chr2,us,df_i=None):
    """
    add all the total licks from a given FNCData object with training
    data to a large dataframe to store all the data from an experiment

    Inputs
    ------
    a: FNCData
    us: str
        what the unconditioned stimulus was for the CS+ flavor for this exp
        (i.e. glucose or fat)
    chr2: bool
        whether or not these were channelrhodopsin mice
    df_i: pandas.DataFrame
        the dataframe to add the data to. if not specified we'll create a new one
    
    Outputs
    -------
    df: pandas.DataFrame
        the compiled dataframe
    """
    d = a.total_licks.dropna().sort_index(axis=1)
    d.columns = pd.MultiIndex.from_tuples([(0,'-',0), 
                                           (0,'+',1), 
                                           (1,'+',0), 
                                           (1,'-',1), 
                                           (2,'-',0), 
                                           (2,'+',1)], 
                                           names = ['day', 'CS', 'sess'])
    d  = (d.stack(['day', 'CS', 'sess'])
           .rename("total_licks")
           .reset_index())
    d['phase'] = 'training'
    d['US']    = us
    d['Chr2']  = chr2
    d['side']  = 'center'
    d['sex']   = a.setup.loc[d.mouse, 'sex'].values
    d['CS+']   = a.setup.loc[d.mouse, 'CS+'].values
    d['CS-']   = a.setup.loc[d.mouse, 'CS-'].values
    d = d.set_index(['US','Chr2','mouse','phase','day','CS'])

    if df_i is None:            
        return d
    else:
        return pd.concat([df_i,d])

    

def get_training_licks(a, chr2, us,df_i=None, s=10000):
    """
    add all the lick times from a given FNCData object with training
    data to a large dataframe to store all the data from an experiment

    Inputs
    ------
    a: FNCData
    us: str
        what the unconditioned stimulus was for the CS+ flavor for this exp
        (i.e. glucose or fat)
    chr2: bool
        whether or not these were channelrhodopsin mice
    df_i: pandas.DataFrame
        the dataframe to add the data to. if not specified we'll create a new one
    
    Outputs
    -------
    df: pandas.DataFrame
        he compiled dataframe
    """

    if a.total_left_licks.dropna().sum().sum()>0:
        df = a.left_licks.dropna().sort_index(axis=1)
    else:
        df = a.right_licks.dropna().sort_index(axis=1)
    df = (df.stack("cumm_sess")
            .apply(lambda x: pd.Series(x[x>0]))
            .rename_axis("lick_number", axis=1)
            .unstack("cumm_sess")
            .stack("lick_number"))
    df.columns=  pd.MultiIndex.from_tuples([(0,'-',0), 
                                            (0,'+',1), 
                                            (1,'+',0), 
                                            (1,'-',1), 
                                            (2,'-',0), 
                                            (2,'+',1)], 
                                           names = ['day', 'CS', 'sess'])
    df = (df.stack(['day', 'CS', 'sess'])
            .rename("time")
            .to_frame())
    df['sex']   = a.setup.loc[df.index.get_level_values("mouse"),'sex'].values
    df['CS+']   = a.setup.loc[df.index.get_level_values("mouse"),'CS+'].values
    df['CS-']   = a.setup.loc[df.index.get_level_values("mouse"),'CS-'].values
    df['phase'] = 'training'
    df['US']    = [us]*len(df)
    df['Chr2']  = [chr2]*len(df)
    df['side']  = ['center']*len(df)
    
    df = (df.reset_index()
            .set_index(['US','Chr2','mouse','phase',
                        'day','CS','lick_number'])
            .sort_index())

    if df_i is None:
        return df
    else:
        return pd.concat([df_i, df])


def get_testing_licks(a, chr2, us, df_i=None):
    """
    add all the lick times from a given FNCData object with testing
    data to a large dataframe to store all the data from an experiment

    Inputs
    ------
    a: FNCData
    us: str
        what the unconditioned stimulus was for the CS+ flavor for this exp
        (i.e. glucose or fat)
    chr2: bool
        whether or not these were channelrhodopsin mice
    df_i: pandas.DataFrame
        the dataframe to add the data to. if not specified we'll create a new one
    
    Outputs
    -------
    df: pandas.DataFrame
        the compiled dataframe
    """
    
    df = (pd.concat({'left' : a.left_licks.dropna().rename_axis(index = "mouse", columns = "day"), 
                     'right': a.right_licks.dropna().rename_axis(index = "mouse", columns = "day")},
                    names = ['side'])
            .stack("day")
            .apply(lambda x: pd.Series(x[x>0]))
            .rename_axis("lick_number", axis=1)
            .stack("lick_number")
            .rename("time")
            .to_frame()
            .reorder_levels(['mouse','day','side', 'lick_number'])
            .sort_index())

    for mouse in df.index.get_level_values('mouse').unique():
        for day in range(df.index.get_level_values('day').max()+1):
            try:
                df.loc[(mouse, day, a.pside.loc[mouse,day]), 'CS'] = '+'
                df.loc[(mouse, day, a.mside.loc[mouse,day]), 'CS'] = '-'
            except KeyError as e:
                print(e)
        df.loc[mouse,'sex'  ] = a.setup.loc[mouse,'sex']
        df.loc[mouse,'CS+'  ] = a.setup.loc[mouse,'CS+']
        df.loc[mouse,'CS-'  ] = a.setup.loc[mouse,'CS-']
    
    df['Chr2' ] = chr2
    df['US'   ] = us
    df['sess' ] = 0
    df['phase'] = 'test'
    
    df = (df.reset_index()
            .set_index(['US','Chr2','mouse','phase',
                        'day','CS','lick_number'])
            .sort_index())

    if df_i is None:
        return df
    else:
        return pd.concat([df_i, df])

def prune(df, both_us = True, drop = []):
    """
    prune the dataframe to only include mice that
    a. made it to the end of the experiment
    b. licked at least once on the second day of testing
    c. got all infusions

    Inputs
    ------
    df: pd.DataFrame
        output from sequentially running get_training_licks
        and get_testing_licks
    
    Outputs
    -------
    df: pd.DataFrame
        pruned dataframe
    survived_idx: List 
        indices of the dataframe that survived
    """
    # get the mice that made it to the end of testing
    # and licked on the second day of testing
    
    _df = (df.set_index(['US','Chr2','mouse','phase','day','CS'])
             .unstack(['phase','CS','day'])
             .dropna())
    mask = (_df.total_licks.training > 20).all(axis=1) & (_df.total_licks.test > 0).all(axis=1)
    if both_us:
        _df = _df.unstack('US')
        mask = mask.unstack('US').fillna(False).all(axis=1)
        _df = _df.loc[mask].stack('US').reorder_levels([2,0,1])
    else:
        _df = _df.loc[mask]
    _df = _df.stack(['phase','day','CS']).sort_index()
    if len(drop)>0:
        _df = _df.loc[~(_df.index.get_level_values('mouse').isin(drop))]
    survived_idx = _df.index
    
    return _df, survived_idx


def format_final_dataset(ctl_train_g=None, ctl_test_g = None, 
                         ctl_train_f = None, ctl_test_f = None, 
                         stim_train_g=None, stim_test_g = None, 
                         stim_train_f = None, stim_test_f = None,
                         both_us = True, to_prune = True, drop = []):
    """
    """
    
    data_objs = {False: {'glucose': {'train': ctl_train_g,  'test': ctl_test_g}, 
                         'fat'    : {'train': ctl_train_f,  'test': ctl_test_f}},
                 True:  {'glucose': {'train': stim_train_g, 'test': stim_test_g}, 
                         'fat'    : {'train': stim_train_f, 'test': stim_test_f}}}
    df = None
    dft = None
    for chr2 in data_objs:
        for us in data_objs[chr2]:
            for phase in data_objs[chr2][us]:
                if data_objs[chr2][us][phase] is not None:
                    if phase == 'train':
                        df = get_total_training_licks( data_objs[chr2][us][phase], chr2, us,  df_i = df)
                        dft = get_training_licks( data_objs[chr2][us][phase], chr2, us, df_i = dft)
                    if phase == 'test':
                        df = get_total_testing_licks( data_objs[chr2][us][phase], chr2, us, df_i = df)
                        dft = get_testing_licks( data_objs[chr2][us][phase], chr2, us, df_i = dft)
    if to_prune:
        df, survived_idx = prune(df.reset_index(), both_us = both_us, drop = drop)
        dft = (dft.reset_index('lick_number')
                  .loc[survived_idx]
                  .set_index('lick_number', append = True))
    return df, dft
    
def get_pretest_pref(a):
    """
    """
    df = get_total_testing_licks(a, 'NA', 'NA', pretest = True).reset_index()
    df['sess'] = df.day
    df['day' ] = 0
    df = df.reset_index().set_index(['mouse', 'sess','CS']).total_licks.unstack('CS')
    pref = df['+']/(df['+']+df['-'])
    pref.name='pref'
    return pref.reset_index()


def pretest_val_plots(stim_g=None , ctl_g=None, stim_f=None, ctl_f=None):
    """
    """
    fig, ax = plt.subplots(2,2, figsize=(10,8))
    a = {'stim g': stim_g, 
         'ctl g': ctl_g, 
         'stim f': stim_f, 
         'ctl f': ctl_f}
    for i,(k, v) in enumerate(a.items()):
        if v is not None:
            df = get_pretest_pref(v)
            g= sns.pointplot(data=df, x='sess', y='pref', hue='mouse', palette = 'tab20c', ax=ax.flatten()[i])
            g.legend(loc='upper left', bbox_to_anchor = (1,1))
            ax.flatten()[i].set_ylim(0,1.1)
            ax.flatten()[i].plot([0,1],[.5,.5], 'k', ls='--')
            ax.flatten()[i].set_title(k)
    sns.despine()
    fig.tight_layout(pad=2.)