import itertools
import pandas as pd
from V1_classes.OSI_DSI_utils import compute_OSI
from V1_classes.utils import SEMf, contains_character


import numpy as np
from numpy.typing import NDArray
    
#stats functions
def get_mean_sem(phys_rec: NDArray, s_obj, cond : str|None = None):
    """
    Calculate mean and SEM for each stimulus type and save the results.

    Parameters:
    - st_data_obj: instance of the stimulation_data class
    - phys_rec (np.ndarray): Array of physiological recordings.
    - n_it (int, optional): Index specifying which logical dictionary to use. Default is 0.
    - change_existing_dict_files (bool, optional): Flag to change existing dictionary files. Default is True.

    Returns:
    Dict[str, Any]: Dictionary containing mean and SEM values for each stimulus type.
    """

    Mean_SEM_dict = {}
    l_dict = s_obj.data[cond]['logical_dict']
    for key in l_dict.keys():
        stim_rec = s_obj.get_recording(stim_name = key, stim_time = None,cond = cond, phys_rec = phys_rec)
        mean_betw_cells = np.mean(stim_rec, axis = 0)
        Mean = np.mean(mean_betw_cells, axis=0)
        #sem between cells for stimuli that are presented only once
        SEM = SEMf(mean_betw_cells)
        Mean_SEM_dict[key] = np.column_stack((Mean, SEM))

    s_obj.data[cond]['mean_sem'] = Mean_SEM_dict
    
def trace_goodness(phys_rec: NDArray, s_obj = None, cond : str|None = None) -> NDArray:
    """ Calculate a metric indicating the "goodness" of a of the physiological signal.
    Originally defined for 2p data.

    :param phys_rec: The physiological data (n cells x timebins).
    :type phys_rec: NDArray
    :return: goodness metric for each cell
    :rtype: NDArray
    """

    if len(phys_rec.shape)>1:
        qt_25 = np.percentile(phys_rec, 25,axis=1)
        qt_99 = np.percentile(phys_rec, 99,axis=1)
        STDs_Q1 = []
        for i,q25 in enumerate(qt_25):
            traccia = phys_rec[i,:]
            dati_primo_qt =  traccia[(traccia <= q25)]
            STDs_Q1.append(np.std(dati_primo_qt))
        STDs_Q1 = np.array(STDs_Q1)
        goodness = qt_99/STDs_Q1

    else:
        qt_5 = np.percentile(phys_rec, 5)
        qt_95 = np.percentile(phys_rec, 95)
        goodness = (qt_95-qt_5)/qt_5

    # Handle cases where the metric is infinite
    if len(phys_rec.shape)>1:
        goodness[goodness==np.Inf] = 0
    else:
        goodness = 0  if goodness==np.Inf else goodness
    
    if s_obj:
        s_obj.data[cond]['t_goodness'] = goodness
    return goodness
    
    
def get_OSI(phys_rec: NDArray, s_obj, cond : str|None = None):
    """
    Calculate Orientation Selectivity Index (OSI) based on stimulation data and physiological recordings.

    Parameters:
    - s_obj: Stimulation data object.
    - phys_rec (np.ndarray): Physiological recording data.
    - n_it (int): Iteration index.
    - change_existing_dict_files (bool): Flag to indicate whether to change existing dictionary files.

    Returns:
    - Tuple[Dict, pd.DataFrame, Dict]: Tuple containing Increase_stim_vs_pre (DF/F stim vs pre), tuningC_df (average tuning curve for each cell + preferred ori and OSI), and Cell_ori_tuning_curve_sem.
    """
    #averaging_window può anche essere settato come intero, che indichi il numero di frame da consconderare
    l_dict = s_obj.data[cond]['logical_dict']
    s_time = s_obj.analysis_settings['stim_duration']
    l = s_obj.analysis_settings['latency']
    
    #Ori: contains numeric, but not literals or +/-
    k_ori = [key for key in l_dict.keys() if contains_character(key, r'\d') and 
            not (contains_character(key, r'[a-zA-Z]') or contains_character(key, r'[+-]'))]
    
    s_obj.data[cond]['OSI_delta_st_pre'] = {}; s_obj.data[cond]['OSI_tuningC_avg'] = {}
    s_obj.data[cond]['OSI_tuningC_sem'] = {}

    for _, k in enumerate(k_ori): #per ogni orientamento...
        ori_rec = s_obj.get_recording(stim_name = k, phys_rec = phys_rec, cond = cond,
                    stim_time = s_time, get_pre_stim = False, latency=l)
        prestim_rec = s_obj.get_recording(stim_name = k, phys_rec = phys_rec, cond = cond,
                    stim_time = s_time, get_pre_stim = True, latency=l)
        
        avg_st = np.mean(ori_rec, axis = 2); avg_pre = np.mean(prestim_rec, axis = 2)

        s_obj.data[cond]['OSI_delta_st_pre'][k] = (avg_st-avg_pre)/avg_pre
        s_obj.data[cond]['OSI_tuningC_avg'][k] = np.nanmean(s_obj.data[cond]['OSI_delta_st_pre'][k],axis=0); 
        s_obj.data[cond]['OSI_tuningC_sem'][k] = SEMf(s_obj.data[cond]['OSI_delta_st_pre'][k])

    s_obj.data[cond]['delta_gray_avg'] = np.mean([np.mean(v, axis = 0)*100 for _,v 
                            in s_obj.data[cond]['OSI_delta_st_pre'].items()], axis=0)
    s_obj.data[cond]['OSI_tuningC_df'] = compute_OSI(s_obj.data[cond]['OSI_tuningC_avg'])  

def get_corr(phys_rec, s_obj, cond : str|None = None):
    """
    Calculate correlation matrix for each stimulus phase and save the results.

    Parameters:
    - phys_rec (np.ndarray): Array of physiological recordings.
    - s_obj: Stimulation data object.
    - cond (str, optional): Condition string. Default is None.
    """
    corr_phases = s_obj.analysis_settings['corr_phases']
    #save the indexes of the upper triangle of the correlation matrix
    upper_tri_idxs = np.triu_indices(phys_rec.shape[0], k=1)
    row_col = [(i,j) for i,j in zip(*upper_tri_idxs)]
    corr_df = pd.DataFrame({'row_col': row_col})
    for phase in corr_phases:
        stim_rec = s_obj.get_recording(stim_name = phase, stim_time = None,cond = cond, phys_rec = phys_rec)
        corr_mat = np.corrcoef(stim_rec)
        corr_vec = corr_mat[upper_tri_idxs]
        corr_df[phase] = corr_vec
    s_obj.data[cond]['corr'] = corr_df  
    
    
def get_corr_dict(stim_data, grouping = 'all'):
    corr_dict = {}
    n_cells = stim_data.recap_stats['pre'].shape[0]
    for cond in stim_data.conditions:
        corr_df = stim_data.data[cond]['corr']
        if grouping == 'all':
            idxs = np.arange(0, n_cells)
            selected_rows = corr_df
        else:
            idxs = sorted(stim_data.rstats_dict[cond][grouping]['idxs_above_threshold'])
            combinations = list(itertools.combinations(idxs, 2))
            selected_rows = corr_df[corr_df['row_col'].isin(combinations)]
            
        stats = []
        for i in idxs:
            sel_df = selected_rows[selected_rows['row_col'].apply(lambda x: i in x)]
            stats.append(np.mean(sel_df.select_dtypes(include=[np.number]), axis = 0))
        df = pd.concat([s.to_frame().T for s in stats])
        df.index = idxs
        corr_dict[cond] = df
    
    return corr_dict
    



