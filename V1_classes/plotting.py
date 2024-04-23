from ast import Tuple
import copy
from os import path
from matplotlib import pyplot as plt

import numpy as np
from numpy.typing import NDArray

from pandas import DataFrame
from typing import Dict, cast
from matplotlib.axes import Axes

from V1_classes.plotting_general import set_default_matplotlib_params
from V1_classes.utils import contains_character

    
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