from argparse import ArgumentParser
from os import path
import os
import platform

from V1_classes.stimulation_handler import multi_session_data, stimulation_data
from V1_classes.utils import Stim_var_rename, add_keys_logicalDict, load_2p_data, read_json, remove_dirs


SCRIPT_DIR     = path.abspath(path.join(__file__,'..'))
print(SCRIPT_DIR)
LOCAL_SETTINGS = path.join(SCRIPT_DIR, 'local_settings.json')
ANALYSIS_SETTINGS  = path.join(SCRIPT_DIR, 'analysis_settings.json')

def path_os_style(fp, splitter = '/', rdir = 'c:'):
    op_sys = platform.system()
    p = path.join(*fp.split(splitter))
    if op_sys == 'Windows':
        p = p.replace(rdir,rdir+os.sep)
    elif op_sys == 'Linux':
        p = os.sep+p
    return p

def main(fp = 'local'):
    script_settings    = read_json(path=LOCAL_SETTINGS)
    analysis_settings  = read_json(path=ANALYSIS_SETTINGS)
    # Set paths as defaults
    if fp == 'local':
        data_root = path_os_style(script_settings["data_root_local"])
    else:
        data_root = path_os_style(script_settings["data_root_drive"])
    data_root = path_os_style(script_settings["data_root_local"])
    multiexp_fp = path_os_style(script_settings["multiexp_fp"])
    #instantiate multiexp
    multiexp = multi_session_data(analysis_settings = analysis_settings,savepath=multiexp_fp)  
    
    #list all subjects present in the data folder
    sbjs = [d.strip() for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]  
    #TO BE CHANGED
    st = analysis_settings["sessions"] #change with the appropriate field in analysis settings
    #if "all" is indicated, gather data from all sessions and all subjects
    if st=='all':
        st = ';'.join([s+'=[]' for s in sbjs])
    #subjects to analyze are separated by semicolumn (;)
    sbjs2an = st.split(';')
    
    # -- ANALYSIS LOOP --
    for sbj in sbjs2an:
        #parsing of the input string (sbj = [sessions])
        id, sessions = [el.strip() for el in sbj.split('=')]
        if id not in sbjs:
            raise ValueError(f"Sbj {id} not present in local data")
        #list all available sessions for subject sbj
        id_available_sess = [d for d in os.listdir(path.join(data_root,id)) if os.path.isdir(os.path.join(data_root,id,d))]
        #sessions are parsed between [] divided by ,. if [] all sessions are selected
        sessions = id_available_sess if sessions == '[]' else sessions.strip('[]').split(',')
        if not(all(s in id_available_sess for s in sessions)):
            raise ValueError(f" Error in naming -Sessions selected for Sbj {id}")
        #the true data analysis is performed within this loop
        for session in sessions:
            try:
                session_fp = path.join(data_root,id,session)
                os.chdir(session_fp)
                #instantiate a stimulation_data object
                stim_data = stimulation_data(fp = session_fp, analysis_settings = analysis_settings,
                                            add_keys_logicalDict= add_keys_logicalDict,
                                            Stim_var_rename = Stim_var_rename)
                if not(analysis_settings['reanalysis']):
                    stim_data.load()
                if analysis_settings['reanalysis'] or  not(stim_data.recap_stats):
                    remove_dirs(fp = session_fp, dirs2rm =['Analyzed_data','Plots'])
                    #load the calcium data
                    F_ds = load_2p_data(analysis_settings)
                    #select the fluorescence of interest (e.g. with or without neuropil)
                    recording = F_ds[analysis_settings['signal_of_interest']]
                    #get the relevant stats from calcium data and save and/or plot them
                    stim_data.get_stats(recording)
                    stim_data.save()
                    #if analysis_settings["recap_stats_plot"]:
                        #stim_data.plot()
                    return stim_data, recording
                #append stats to multiexp
                multiexp.append_stats(stim_data.recap_stats, sbj = id, sess = session)
                print(f" Session {session} of sbj {id} analyzed successfully")
                
            except:
                print(f" Unable to analyze session {session} of sbj {id}")
        
        #get stats for multiexp data and save them
        multiexp.get_stats(threshold_dict=analysis_settings["threshold_dict"])
        multiexp.save()
        return multiexp
        
        
        
if __name__ == '__main__':
    main()