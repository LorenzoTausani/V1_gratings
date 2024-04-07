from collections import defaultdict
from datetime import datetime
import json
from os import path
import os
import pickle
import re
import warnings
from scipy.stats import mode
from typing import Any, Callable, Dict
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from pandas import DataFrame
from V1_classes.plotting import recap_stats_plot
from V1_classes.stat_funs import *

from V1_classes.utils import find_files_by_extension, get_relevant_cell_stats

#functions that will be computed on 2p data
statf_dict = {'mean': get_mean_sem,
            'goodness': trace_goodness,
            'OSI': get_OSI}


class stimulation_data:
    def __init__(self, fp: str, analysis_settings:dict[str,Any], 
                add_keys_logicalDict: Callable[[dict[str, NDArray]],dict[str, NDArray]] | None = None,
                Stim_var_rename: Callable[[DataFrame,str],DataFrame] | None = None,
                stats_fun: list[Callable[['stimulation_data',str,NDArray],None]] = [get_mean_sem]):

        """
        Class to handle stimulation data coming from file .xlsx

        Parameters:
        :param fp: directory containing the .xlsx file
        :type fp: str
        :param analysis_settings: dictionary containing the info for the analysis to be carried out
        :type analysis_settings: dict[str,Any]
        :param add_keys_logicalDict: function for adding integrated keys (e.g. '+') to the logical dictionary
        :type add_keys_logicalDict: Callable[[dict[str, NDArray]],dict[str, NDArray]] | None
        :param Stim_var_rename: function to rename the variables used in the stimulation (e.g. '180.0' -> '180)
        :type Stim_var_rename: Callable[[DataFrame,str],DataFrame] | None
        :param stats_fun: list of functions to be applied to the physiological data. Currently unused.
        :type stats_fun: list[Callable[['stimulation_data',str,NDArray],None]]        
        """
        
        self.path = fp
        self.analysis_settings = analysis_settings
        self.stim_var = analysis_settings['stim_var']
        self.conditions = analysis_settings['conditions']
        if self.conditions==[]:
            self.conditions = ['exp']
        self.recap_stats_plot = analysis_settings['recap_stats_plot']
        self.idxs_only_pre = analysis_settings['idxs_only_pre']
        self.ld_addkeys = add_keys_logicalDict
        self.df_var_rename = Stim_var_rename
        
        self.data = defaultdict(dict)
        self.get_stim_data()
        self.stats_fun = [statf_dict[k] for k in analysis_settings['stats_fun'] if k in statf_dict]
        self.recap_stats = defaultdict(dict)
        self.rstats_dict = defaultdict(dict)

    def get_stim_data(self):
        """Processing of stimulation data.
        It creates the dict data, that will contain all the elaborated data of our analysis.
        When called, get_stim_data organizes stimulation data (the most important is the key 
        logical_dict, that gathers all timestamps of different stimuli).
        """
        xlsx_files = find_files_by_extension(dir = self.path, extension='.xlsx')
        #session names will contain the file dirs, without the extension
        
        #multiple .xlsx files are considered in case of multiple treatments in the same session (pre, post)
        #TODO: do appropriate ordering
        for ex_f in xlsx_files:
            s_name = path.splitext(path.basename(ex_f))[0] #name of the session
            
            s_df = pd.read_excel(ex_f) #get the session df
            if self.df_var_rename: #do the variable renaming in case it has been initialized
                s_df = self.df_var_rename(s_df, self.stim_var)
                
            #get the session condition
            d_key = [c for c in self.conditions if c in s_name]
            if len(d_key)>1:
                raise TypeError(f"More that one condition found for session {s_name}: {d_key}")
            elif d_key==[]:
                raise TypeError(f"No condition found for session {s_name}")
            d_key = d_key[0]
                
            #gather info about the session
            self.data[d_key]['s_name'] = s_name
            self.data[d_key]['df'] = s_df
            self.data[d_key]['stim_vec'] = self.get_StimVec(s_df)
            self.data[d_key]['stim_len'] = self.get_len_stims(s_df)

            #here below i create the logical dictionary
            stim_names = s_df[self.stim_var].unique() #name of all unique stimuli
            logical_dict = {}
            for stim in stim_names:
                if stim != 'END':
                    #define a boolean vector stimTrue
                    stimTrue = self.data[d_key]['stim_vec'] == stim
                    #convert stimTrue in string of 1(T) and 0(F)
                    stimTrue_01 = ''.join('1' if x else '0' for x in stimTrue)
                    #find all sequences of consecutive 1 in stimTrue_01 and compute 
                    #their beginning and end indexes
                    logical_dict[str(stim)] = np.array([(match.start(), match.end()) 
                                            for match in re.finditer('1+', stimTrue_01)])
            if self.ld_addkeys: #if indicated, add integrated keys
                logical_dict = self.ld_addkeys(logical_dict)
                
            self.data[d_key]['logical_dict'] = logical_dict

            
    def get_StimVec(self, stimulation_df: DataFrame) -> NDArray:
        """
        Get a 1d vector that contains, for each time bin, the type of stimulation used.
        :param stimulation_df: dataframe containing stimulation data
        :return: stimuls vector
        :rtype: NDArray
        """
        time_var = self.analysis_settings['time_var']; stim_var = self.analysis_settings['stim_var']
        #initialize a vector as long as the frames of the experiment
        StimVec = np.empty(stimulation_df[time_var].max(), dtype=object) 
        #fill in StimVec with the appropriate labels
        top=0 #pointer
        for it, row in stimulation_df.iterrows(): #for every row of the df...
            if it==0:
                prec_row = row
            else: #if it>0, fill the entries the pointer and the end of that stim (row[time_var])
                # with the name of the stimulus itself (i.e. prec_row[stim_var])
                StimVec[top:row[time_var]] = prec_row[stim_var]
                top=row[time_var]
                prec_row = row
        return StimVec
    
    def get_len_stims(self, stimulation_df: DataFrame): #controlla per dati diversi da Fluorescenza 2p
        """
        Get len of the recording from stimulation data
        :param stimulation_df: dataframe containing stimulation data
        :type stimulation_df: DataFrame
        :return: length of the recording
        :rtype: List[Union[int, float]
        """
        return stimulation_df[self.analysis_settings['time_var']].iloc[-1]
    
    #@property
    #def len_recording(self)      -> NDArray: return np.stack(self._codes)
    
    def get_recording(self, stim_name: str|list[str], phys_rec: NDArray, cond: str = 'exp',
                    stim_time: int|str|None = None, get_pre_stim: bool = False, latency: int=0) -> NDArray:
        """
        Retrieves the physiological recordings corresponding to each occurrence of a stimulus.

        Parameters:
        :param stim_name: name of the stimulus/stimuli of interest.
                        shortcuts of interest include: 'stim', 'gray', 'stim_gray','spontaneous', 'all'
        :type stim_name: str|list[str]
        :param phys_rec: array containing physiological data
        :type phys_rec: NDArray
        :param cond: session on which to work (pre/psilo)
        :type cond: str
        :param stim_time: duration of the stimulus of interest. If None, it reports all indexes
            where the stims in stim_name were present. if 'settings', it takes the stim_time indicated in
            the analysis_settings.
            NOTE: if stim_time = None, you will get in output a 2D matrix (i.e. cells x timebin), else you will 
            get a 3D matrix (i.e. stim occurrence x cells x timebin)
        :type stim_time: int|None
        :param get_pre_stim: flag to decide wether to take the interval of interest
                            before the stimulus onset
        :type get_pre_stim: bool
        :param latency: session on which to work (pre/psilo)
        :type latency: int
        
        :return: Array containing the stimulus' physiological recordings.
        :rtype: NDArray
        """
        #short names for particular parts of the session
        if stim_name=='stim':
            stim_name = ['+','-']
        elif stim_name=='gray':
            stim_name = ['gray +', 'gray -']
        elif stim_name == 'stim_gray':
            stim_name = ['+','-','gray +', 'gray -']
        elif stim_name == 'spontaneous':
            stim_name = ['initial gray','final gray']
        elif stim_name == 'all':
            stim_name = ['initial gray','final gray','+','-','gray +', 'gray -']

        #convert stim_name into a list if it was given as a single string
        stim_name = [stim_name] if isinstance(stim_name, str) else stim_name
        
        if stim_time == 'settings':
            stim_time = self.analysis_settings['stim_duration']
        
        logic_dict = self.data[cond]['logical_dict']
        #get the intervals where the stimulus is on (stim on) and their durations (stim_durations)
        stim_on    = np.concatenate([logic_dict[k] for k in stim_name])
        stim_on    = stim_on[np.argsort(stim_on[:, 0])] #sort rows according to the first element of each row
        stim_durations = stim_on[:, 1] - stim_on[:, 0]
        # -- CASE 1: STIM TIME IS INDICATED (OUT: stim occurrence x cells x timebin) --
        if stim_time:
            #'mode': we assume as stim duration the mode of the durations present in logical dict
            stim_time = int(mode(stim_durations)[0]) if stim_time == 'mode' else int(stim_time)
            #initialize the NDArray containing the recordings (stim_phys_rec)
            stim_phys_rec = np.full((stim_on.shape[0], phys_rec.shape[0],stim_time), np.nan)
            for i, stim_ev in enumerate(stim_on):
                sev_begin = stim_ev[0]
                #arbitrary criterion to assert that durations are correct
                is_duration_correct = np.abs(stim_durations[i]-int(mode(stim_durations)[0]))< int(mode(stim_durations)[0])/10 
                if not(is_duration_correct):
                    warnings.warn('stimulus nr '+ str(i) + 'session '+ cond +
                                'is of different length', UserWarning)
                #check that the stim_ev has been fully recorded physiologically
                is_phys_registered = phys_rec.shape[1] >= stim_on[i, 1]
                if not(is_phys_registered):
                    warnings.warn('stimulus nr '+ str(i) + 'session '+ cond +
                                'was not fully recorded', UserWarning)
                if is_duration_correct and is_phys_registered:
                    if get_pre_stim:
                        sev_begin = sev_begin-latency
                        stim_phys_rec[i,:,:] = phys_rec[:,sev_begin-stim_time:sev_begin]
                    else:
                        sev_begin = sev_begin+latency
                        stim_phys_rec[i,:,:] = phys_rec[:,sev_begin:sev_begin+stim_time]
        # -- CASE 2: STIM TIME IS None (OUT: cells x timebin) --
        else:
            #get the idxs of the timebins where the stimulus/i of interest were present
            idxs = np.concatenate([np.arange(start, end) for start, end in stim_on])
            stim_phys_rec = phys_rec[:,idxs]
            
        return stim_phys_rec


    def get_stats(self, phys_rec: NDArray, recap_vars = ['t_goodness', 'delta_gray_avg', 'OSI_tuningC_df']):
        """get the statistics you are interested in from your physiological recordings

        :param phys_rec: array containing physiological data
        :type phys_rec: NDArray
        :param recap_vars: statistics you want to gather, 
            defaults to ['t_goodness', 'delta_gray_avg', 'OSI_tuningC_df']
        :type recap_vars: list, optional
        """
        #TODO: la logica di estrazione delle sessioni va cambiata per il caso di sessioni singole
        end_conditions = {c:self.data[c]['stim_len'] for c in self.conditions}

        # -- COMPUTE THE STATS ON THE RECORDING FOR EACH CONDITION --
        for fun in self.stats_fun: #for each of the stats_fun indicated during init... 
            for k,data in self.data.items():
                #slice the recording belonging to the particular condition
                if k==self.conditions[0]:
                    #ASSUMPTION: the recording starts with the experiment (this is the case for 2p)
                    #to be checked for new data
                    rec = phys_rec[:,:data['stim_len']]
                else:
                    #each phase begins from the end of the previous one
                    start_rec = end_conditions[previous_key]
                    rec = phys_rec[:,start_rec:start_rec+data['stim_len']]
                previous_key = k
                fun(rec,self,cond = k)
                
        # -- GATHER THE STATISTICS OF INTEREST --  
        for k,v in self.data.items():
            concat_dfs = []
            for rv in recap_vars:
                if rv=='OSI_tuningC_df':
                    concat_dfs.append(v[rv].loc[:, ['Preferred or', 'OSI']])
                else:
                    concat_dfs.append(pd.DataFrame({rv: v[rv]}))
            self.recap_stats[k] = pd.concat(concat_dfs, axis=1)
            self.rstats_dict[k] = get_relevant_cell_stats(self.recap_stats[k], 
                                    self.analysis_settings["threshold_dict"])
            
    def save(self):
        exp_n = path.splitext(path.basename(self.path))[0]
        fn = "sd_"+exp_n+".pkl"
        os.chdir(self.path)
        os.makedirs('Analyzed_data', exist_ok=True)
        os.chdir(path.join(self.path,'Analyzed_data'))
        with open(fn, "wb") as file:
            pickle.dump(self, file)
            
    def load(self):
        exp_n = path.splitext(path.basename(self.path))[0]
        fn = "sd_"+exp_n+".pkl"
        os.chdir(self.path)
        if path.exists(path.join(self.path,'Analyzed_data',fn)):
            os.chdir(path.join(self.path,'Analyzed_data'))
            with open(fn, "rb") as file:
                loaded = pickle.load(file)
                self.__dict__.update(loaded.__dict__)
            print(f"file {fn} loaded successfully")
        else:
            print(f"file {fn} not found")
    
    def plot(self):
        os.chdir(self.path)
        os.makedirs('Plots', exist_ok=True)
        os.chdir('Plots')
        for grouping in self.recap_stats_plot:
            os.makedirs(grouping, exist_ok=True)
            if grouping == 'all':
                idxs = {'pre': None,'psilo': None}
            else:
                if self.idxs_only_pre:
                    idxs = self.rstats_dict['pre'][grouping]['idxs_above_threshold']
                    idxs = {k:idxs for k in self.rstats_dict.keys()}
                else:
                    idxs = {k:v[grouping]['idxs_above_threshold'] for k,v in self.rstats_dict.items()}
            for k in self.recap_stats[self.conditions[0]].keys():
                x_range = (0,1) if k=='OSI' else None
                    
                recap_stats_plot(self.recap_stats,         
                    var = k,
                    x_range = x_range, 
                    idxs = idxs,
                    out_dir = path.join(self.path,'Plots',grouping)
                    )

class multi_session_data:
    def __init__(self, analysis_settings:Dict[str,Any], savepath:str) -> None:
        #get current datetime. It will be the name of the analysis folder
        mexp_datetime = datetime.now().strftime("%Y-%m-%d %H_%M_%S")
        self.path = path.join(savepath,mexp_datetime)
        self.analysis_settings = analysis_settings
        self.conditions = analysis_settings['conditions']
        if self.conditions == []:
            self.conditions = 'exp'
        
        self.sessions = defaultdict(list)
        
    def append_stats(self, recap_stats, sbj,sess):

        for k,v in recap_stats.items():
            v['Sbj'] = [sbj]*v.shape[0]; v['Session'] = [sess]*v.shape[0] 
            self.sessions[k].append(v)
                
    def get_stats(self, threshold_dict):
        self.data = {k:pd.concat(v, axis=0) for k,v in self.sessions.items()}
        self.rstats_dict = {k:get_relevant_cell_stats(self.data[k], 
                                        threshold_dict) for k in self.data.keys()}
        
    def select_data(self, pre_idxs = False, sel_key = 'delta_gray_avg'):
        if pre_idxs:
            idxs = {k:self.rstats_dict[self.conditions[0]][sel_key]['idxs_above_threshold'] for k in self.rstats_dict.keys()}
        else:
            idxs = {k:v[sel_key]['idxs_above_threshold'] for k,v in self.rstats_dict.items()}
        
        return {k:v.iloc[idxs[k]] for k,v in self.data.items()}
        
    def save(self):
        os.makedirs(self.path, exist_ok=True)
        os.chdir(self.path)
        os.makedirs('Analyzed_data', exist_ok=True)
        os.chdir(os.path.join(self.path,'Analyzed_data'))
        fn = "mexp_data.pkl"
        with open(fn, "wb") as file:
            pickle.dump(self, file)
        with open('params.json', 'w') as json_file:
            # Convert the dictionary to JSON and write it to the file
            json.dump(self.analysis_settings, json_file , indent=4)
        
        
        

