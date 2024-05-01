from ast import Tuple
import copy
from os import path
from matplotlib import pyplot as plt

import numpy as np
from numpy.typing import NDArray

from pandas import DataFrame
from typing import Dict, cast, Callable
from matplotlib.axes import Axes

from V1_classes.plotting_general import get_color_from_name, get_colors, set_default_matplotlib_params
from V1_classes.utils import SEMf, contains_character

    
def recap_stats_plot(sd_recap_stats,
            var: str ='OSI',
            ax: Axes|None = None, 
            x_range: tuple | None = None, 
            num_bins: int  = 25,
            style: Dict[str, Dict[str, str]] = {
                        'pre': {'col': 'green'},
                        'psi': {'col': 'purple'}},
            idxs: Dict[str, NDArray | None] = {'pre': None,'psi': None},
            bar_width: float = 0.4,
            out_dir: str|None = None
            ):
    """Plot summarizing recap stats (bar plot for cat vars, 
    hist for quantitative vars)

    :param sd_recap_stats: recap stats of the session of interest
    :type sd_recap_stats: _type_
    :param var: variable of sd_recap_stats to be plotted, defaults to 'OSI'
    :type var: str, optional
    :param ax: Axis object for subplots, defaults to None
    :type ax: Axes | None, optional
    :param x_range: range lims of the var of interest, defaults to None
    :type x_range: tuple | None, optional
    :param num_bins: num bins in the histogram, defaults to 25
    :type num_bins: int, optional
    :param style: plot styling params
        , defaults to { 'pre': {'col': 'green'}, 'psi': {'col': 'purple'}}
    :type style: Dict[str, Dict[str, str]], optional
    :param idxs: subselection of indexes for sd_recap_stats, 
        defaults to {'pre': None,'psi': None}
    :type idxs: Dict[str, NDArray | None], optional
    :param bar_width: width of bar in barplot, defaults to 0.4
    :type bar_width: float, optional
    :param out_dir: directory where to save the data, defaults to None
    :type out_dir: str|None, optional
    """
    
    set_default_matplotlib_params(l_side  = 15, shape   = 'rect_wide')
    #i create a deepcopy of the sd_recap_stats dataset in order
    #to avoid rewriting sd_recap_stats
    ds_deepcopy = copy.deepcopy(sd_recap_stats)
    #select the indexes of interest
    for k,v in idxs.items():
        if v is not None:
            ds_deepcopy[k] =  ds_deepcopy[k].iloc[v]

    if not(ax):
        fig, ax = plt.subplots(1)

    # Compute min and max values
    if not x_range:
        x_range = (min([v[var].min() for _,v in ds_deepcopy.items()]),
                max([v[var].max() for _,v in ds_deepcopy.items()]))    
                
    
    for i,(k,v) in enumerate(ds_deepcopy.items()):
        if v[var].dtype == 'object': #i.e. it is composed of strings
            fn = 'bar_'+var+'.png'
            counts = v[var].value_counts(normalize=True) * 100
            #sort by orientation
            counts.index = counts.index.astype(int)
            counts = counts.sort_index()
            
            bar_pos = range(len(counts)) if i==0 else [p + bar_width for p in bar_pos]
            ax.bar(bar_pos,
                        counts.values,
                        color = style[k]['col'],
                        edgecolor=style[k]['col'],
                        label = k.capitalize(),
                        width = bar_width)
            
            ax.axhline(y=100/len(counts), color='r', linestyle='--')
            
            if i==0:
                ax.set_xticks([p + bar_width / 2 for p in bar_pos], counts.index)
                ax.set_ylabel('%')
        else:
            fn = 'hist_'+var+'.png'
            ax.hist(
                v[var],
                bins=num_bins,
                range=x_range,
                density=True,
                color = style[k]['col'],
                edgecolor=style[k]['col'],
                alpha=0.5,
                label = k.capitalize()
            )
            
            ax.set_ylabel('Prob. density')
    
    # Plot names
    ax.set_xlabel(var.capitalize())
    ax.legend()
    if out_dir:
        out_fp = path.join(out_dir, fn)
        fig.savefig(out_fp, bbox_inches="tight")
    
def plot_mean_pm_sem(data: NDArray, ax: Axes|None = None, color: str = 'red', line_lab: str = 'Avg') -> tuple[NDArray, NDArray]:
    """
    Plot the mean +/- standard error of the mean (SEM) of the given data.

    Parameters:
        data (NDArray): The input data as a numpy array.
        ax (Axes, optional): The axis to plot on. If None, a new figure is created. Defaults to None.
        color (str, optional): The color of the plot. Defaults to 'red'.

    Returns:
        tuple[NDArray, NDArray]: A tuple containing the mean and SEM arrays.
    """
    if ax is None:
        fig, ax = plt.subplots()
    params = set_default_matplotlib_params(l_side=15, shape='rect_wide')

    mean = np.mean(data, axis=0); sem = SEMf(data, axis=0)
    upper_bound = mean + sem; lower_bound = mean - sem

    ax.plot(mean, color=color, label=line_lab)
    ax.fill_between(range(len(mean)), lower_bound, upper_bound, color=color, alpha=0.2)

    if ax is None:
        plt.show()
    
    return mean, sem

def plot_PSTH(stim_data, phys_rec: NDArray, cells_of_interest: dict[str,NDArray],  
            out_dir: str|None = None, stimuli_of_interest: list|None = None, time_stim: int = 150, 
            time_prestim: int = 150 ,ylbl: str = 'Fluorescence', xlbl: str = 'time(frames)',
            grouping_func: Callable|None = None) -> NDArray:
    """
    Plot the Peri-Stimulus Time Histogram (PSTH) for a given physiological recording over a set of stimuli.

    Parameters:
        stim_data: The stimulus data object.
        phys_rec: The physiological recording data.
        stimuli_of_interest: The list of stimuli of interest.
        cells_of_interest: The indices of cells of interest.
        cond: The index for logical dictionary. Defaults to 0.
        time_stim: The duration of the stimulus. Defaults to 150.
        time_prestim: The duration of the pre-stimulus. Defaults to 150.
        ylbl: y label
        xlbl: x label

    Returns:
        NDArray: The concatenated pre-post stimulus data.
    """
    if stimuli_of_interest is None: 
        all_stims = list(stim_data.data['pre']['logical_dict'].keys())
        #we exclude grays and spontaneous activity
        stimuli_of_interest = [st for st in all_stims if contains_character(st,'[\d\+\-]') and not(contains_character(st,'g'))]
        def extract_number(s):
            numeric_part = ''.join(filter(str.isdigit, s))
            if '+' in s:
                sign = -0.1
            elif '-' in s:
                sign = -0.2
            else:
                sign = 0
            numeric = int(numeric_part) if numeric_part else -1
            return numeric + sign
        stimuli_of_interest.sort(key=lambda x: extract_number(x))
        
    conds = list(stim_data.data.keys())
    if grouping_func is not None:
        stimuli_of_interest, groupnames = grouping_func(stimuli_of_interest)
        cols = get_colors(max(len(st) for st in stimuli_of_interest))
        line_labs = ['-','+','all']
    else:
        groupnames = None
        cols = [get_color_from_name('red')]
    params = set_default_matplotlib_params(l_side=15, shape='rect_wide'); subplots_nr = len(stimuli_of_interest)
    
    fig, axes = plt.subplots(subplots_nr, len(conds), figsize=(params['figure.figsize'][0], params['figure.figsize'][1] * subplots_nr))  # Create subplot grid
    for c_i,cond in enumerate(conds):
        cells = cells_of_interest[cond]
        for i, stimulus in enumerate(stimuli_of_interest):
            if not isinstance(stimulus, list): 
                groupname = stimulus
                stimulus = [stimulus]
            else:
                groupname = groupnames[i]

            for j, stim in enumerate(stimulus):
                try:
                    #selection of stimulus-related data
                    stimulus_phys_rec = stim_data.get_recording(stim, phys_rec, cond = cond, stim_time=time_stim)
                    selected_stim_recs = stimulus_phys_rec[:, cells, :]
                    if not(isinstance(cells, int)):
                        selected_stim_recs = selected_stim_recs.reshape(-1, selected_stim_recs.shape[-1])   
                    if time_prestim > 0: #If requested, selection of period preceding the stimulus
                        pre_phys_recs = stim_data.get_recording(stim, phys_rec, cond = cond, get_pre_stim=True, stim_time=time_stim)
                        selected_pre_recs = pre_phys_recs[:, cells, :]
                        if not(isinstance(cells, int)):
                            selected_pre_recs = selected_pre_recs.reshape(-1, selected_pre_recs.shape[-1])
                        pre_post_stim = np.hstack((selected_pre_recs, selected_stim_recs))
                        stim_onset = selected_pre_recs.shape[-1]; axes[i,c_i].axvline(x=stim_onset, color='green', linestyle='--') #line indicating stimulus onset
                    else:
                        pre_post_stim = selected_stim_recs
                    line_lab = 'all' if stim[-1].isdigit() else stim[-1]
                    plot_mean_pm_sem(pre_post_stim, ax=axes[i,c_i], color = cols[j], line_lab = line_lab)
                except:
                    print(f"Error in plotting {stim} {cond}") 
                    
            axes[i,c_i].set_title(groupname+' '+cond); axes[i,c_i].set_xlabel(xlbl); axes[i,c_i].set_ylabel(ylbl)
            lab_col_style = [[l,c] for l,c in zip(line_labs, cols)]
            arbitrary_legend(fig, lab_col_style)

    plt.tight_layout()
    plt.show()
    if out_dir:
        out_fp = path.join(out_dir, 'PSTH.png')
        fig.savefig(out_fp, bbox_inches="tight")
    
    return pre_post_stim

def Ori_pieplot(out_df: DataFrame, 
                ax: Axes | None = None):
    """_summary_

    :param out_df: _description_
    :type out_df: DataFrame
    :param ax: _description_, defaults to None
    :type ax: Axes | None, optional
    """
    set_default_matplotlib_params(l_side  = 15, shape   = 'square')
    if not(ax):
        fig, ax = plt.subplots(1)
    ax = cast(Axes, ax)
    #get the frequency of each preferred orientation
    preferred_or_counts = out_df['Preferred or'].value_counts()
    preferred_or_counts.index = preferred_or_counts.index.astype(int)
    preferred_or_counts = preferred_or_counts.sort_index()
    
    jet_colors = plt.cm.jet(np.linspace(0, 1, len(preferred_or_counts)//2))
    #use the same color for the same orientation
    jet_colors = np.tile(jet_colors, (2, 1))
    preferred_or_counts.plot(kind='pie', autopct='%1.1f%%', colors = jet_colors)
    plt.title('Preferred Orientation')
    plt.ylabel('') 
    
'''
def plot_cell_tuning(Tuning_curve_avg_DF: DataFrame, cell_id: int) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the tuning curve for a specific cell.

    Parameters:
    - Tuning_curve_avg_DF: DataFrame containing tuning curve data.
    - cell_id: Index of the cell to plot.

    Returns:
    A tuple containing the Matplotlib Figure and Axes objects (now disenabled)
    """
    params = set_default_matplotlib_params(l_side= 15, shape = 'rect_wide'); fontsz = params['font.size']
    mrkr = '.'; mrkr_sz = params['lines.markersize']
    # Extract orientation columns and index columns
    ori_columns = [c for c in Tuning_curve_avg_DF.columns if not(contains_character(c,r'[a-zA-Z]'))]
    indexes_columns = [c for c in Tuning_curve_avg_DF.columns if contains_character(c,r'[a-zA-Z]')]

    # Extract cell tuning and statistics of that cell
    cell_tuning = Tuning_curve_avg_DF[ori_columns].iloc[cell_id,:]
    cell_stats = Tuning_curve_avg_DF[indexes_columns].iloc[cell_id,:]; cell_stats = round_numeric_columns(cell_stats) #i round cell_stats to 2 digits
    sorted_colnames = sorted(cell_tuning.index.tolist(), key=sort_by_numeric_part) # Sort column names based on numeric part

    fig, ax = plt.subplots()
    
    ax.plot(cell_tuning[sorted_colnames], color='purple', marker=mrkr, markersize=mrkr_sz) # Plot the tuning curve
    ax.set_xlabel('Orientation'); ax.set_ylabel('(Fstim-Fpre)/Fpre'); ax.tick_params(axis='x', labelsize=(fontsz/len(sorted_colnames))*6)
    
    text_content = "\n".join([k+'= '+str(cell_stats[k]) for k in cell_stats.keys()]) # Create text content for textbox
    textbox = ax.text(1.05, 0.5, text_content, transform=ax.transAxes, fontsize=fontsz, verticalalignment='center', bbox=dict(facecolor='white', alpha=0.5)) # Add textbox to the plot

    #return fig, ax
'''