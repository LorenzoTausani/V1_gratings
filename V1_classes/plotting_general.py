import matplotlib
import matplotlib.pyplot as plt 
from matplotlib.axes import Axes
from typing import List, Literal, Dict, Any, TypeVar, Union

import numpy as np
from numpy.typing import NDArray

# --- DEFAULT PLOTTING PARAMETERS ----

_Shapes = Literal['square', 'rect_tall', 'rect_wide']
_ax_selection = Literal['x', 'y', 'xy']

# -- da zdream.utils.misc --
# Type generics
T = TypeVar('T')
D = TypeVar('D')
# Default for None with value
def default(var : T | None, val : D) -> T | D:
    return val if var is None else var

def _get_appropriate_fontsz(xlabels: List[str], figure_width: float | int | None = None) -> float:
    '''
    Dynamically compute appropriate fontsize for xticks 
    based on the number of labels and figure width.
    
    :param labels: Labels in the abscissa ax.
    :type labels: List[str]
    :param figure_width: Figure width in inches. If not specified using default figure width.
    :type figure_width: float | int | None
    
    :return: Fontsize for labels ticks in abscissa ax.
    :rtype: float
    '''
    
    # Compute longest label length
    max_length = max(len(label) for label in xlabels)
    
    # In case where figure_width not specified, use default figsize
    figure_width_: int | float = default(figure_width, plt.rcParams['figure.figsize'][0])
        
    # Compute the appropriate fontsize to avoid overlaps 
    # NOTE: Ratio factor 0.0085 was found empirically
    fontsz = figure_width_ / len(xlabels) / max_length / 0.0085 
    
    return fontsz


def subplot_same_lims(axes: NDArray, sel_axs:_ax_selection = 'xy'):
    """set xlim, ylim or both to be the same in all subplots

    :param axes: array of Axes objects (one per subplot)
    :type axes: NDArray
    :param sel_axs: axes you want to set same bounds, defaults to 'xy'
    :type sel_axs: _ax_selection, optional
    """
    #lims is a array containing the lims for x and y axes of all subplots
    lims = np.array([[ax.get_xlim(), ax.get_ylim()] for ax in axes.flatten()])
    #get the extremes for x and y axis respectively among subplots
    xlim = [np.min(lims[:,0,:]), np.max(lims[:,0,:])] 
    ylim = [np.min(lims[:,1,:]), np.max(lims[:,1,:])]
    #for every Axis obj (i.e. for every subplot) set the extremes among all subplots
    #depending on the sel_axs indicated (either x, y or both (xy))
    #NOTE: we could also improve this function by allowing the user to change the
    #limits just for a subgroup of subplots
    for ax in axes.flatten():
        ax.set_xlim(xlim) if 'x' in sel_axs else None
        ax.set_ylim(ylim) if 'y' in sel_axs else None   
        
        
def set_default_matplotlib_params(
    l_side  : float     = 15,
    shape   : _Shapes   = 'square', 
    xlabels : List[str] = []
) -> Dict[str, Any]:
    '''
    Set default Matplotlib parameters.
    
    :param l_side: Length of the lower side of the figure, default to 15.
    :type l_side: float
    :param shape: Figure shape, default is `square`.
    :type shape: _Shapes
    :param xlabels: Labels lists for the abscissa ax.
    :type xlabels: list[str]
    
    :return: Default graphic parameters.
    :rtype: Dict[str, Any]
    '''
    
    # Set other dimension sides based on lower side dimension and shape
    match shape:
        case 'square':    other_side = l_side
        case 'rect_wide': other_side = l_side * (2/3)
        case 'rect_tall': other_side = l_side * (3/2)
        case _: raise TypeError(f'Invalid Shape. Use one of {_Shapes}')
    
    # Default params
    fontsz           = 35
    standard_lw      = 4
    box_lw           = 3
    box_c            = 'black'
    subplot_distance = 0.3
    axes_lw          = 3
    tick_length      = 6
    
    # If labels are given compute appropriate fontsz is the minimum
    # between the appropriate fontsz and the default fontsz
    if xlabels:
        fontsz =  min(
            _get_appropriate_fontsz(xlabels=xlabels, figure_width=l_side), 
            fontsz
        )
    
    # Set new params
    params = {
        'figure.figsize'                : (l_side, other_side),
        'font.size'                     : fontsz,
        'axes.labelsize'                : fontsz,
        'axes.titlesize'                : fontsz,
        'xtick.labelsize'               : fontsz,
        'ytick.labelsize'               : fontsz,
        'legend.fontsize'               : fontsz,
        'axes.grid'                     : False,
        'grid.alpha'                    : 0.4,
        'lines.linewidth'               : standard_lw,
        'lines.linestyle'               : '-',
        'lines.markersize'              : 20,
        'xtick.major.pad'               : 5,
        'ytick.major.pad'               : 5,
        'errorbar.capsize'              : standard_lw,
        'boxplot.boxprops.linewidth'    : box_lw,
        'boxplot.boxprops.color'        : box_c,
        'boxplot.whiskerprops.linewidth': box_lw,
        'boxplot.whiskerprops.color'    : box_c,
        'boxplot.medianprops.linewidth' : 4,
        'boxplot.medianprops.color'     : 'red',
        'boxplot.capprops.linewidth'    : box_lw,
        'boxplot.capprops.color'        : box_c,
        'axes.spines.top'               : False,
        'axes.spines.right'             : False,
        'axes.linewidth'                : axes_lw,
        'xtick.major.width'             : axes_lw,
        'ytick.major.width'             : axes_lw,
        'xtick.major.size'              : tick_length,
        'ytick.major.size'              : tick_length,
        'figure.subplot.hspace'         : subplot_distance,
        'figure.subplot.wspace'         : subplot_distance
    }
    
    plt.rcParams.update(params)

    return params

# --- STYLING --- 

def customize_axes_bounds(ax: Axes):
    '''
    Modify axes bounds based on the data type.

    For numeric data, the bounds are set using the values of the second and second-to-last ticks.
    For categorical data, the bounds are set using the indices of the first and last ticks.

    :param ax: Axes object to be modified.
    :type ax: Axes
    '''
    
    # Get tick labels
    tick_positions_x = ax.get_xticklabels()
    tick_positions_y = ax.get_yticklabels()
    
    # Compute new upper and lower bounds for both axes
    t1x = tick_positions_x[ 1].get_text()
    t2x = tick_positions_x[-2].get_text()
    t1y = tick_positions_y[ 1].get_text()
    t2y = tick_positions_y[-2].get_text()
    
    # Set new axes bound according to labels type (i.e. numeric or categorical)
    for side, (low_b, up_b) in zip(['left', 'bottom'], ((t1y, t2y), (t1x, t2x))):
        
        # Case 1: Numeric - Set first and last thicks bounds as floats
        if low_b.replace('−', '').replace('.', '').isdigit():
            low_b = float(low_b.replace('−','-'))
            up_b  = float( up_b.replace('−','-'))
            
            # Remove ticks outside bounds
            ticks = tick_positions_y if side == 'left' else tick_positions_x
            ticks = [float(tick.get_text().replace('−','-')) for tick in ticks 
                     if low_b <= float(tick.get_text().replace('−','-')) <= up_b]
            
            ax.set_yticks(ticks) if side == 'left' else ax.set_xticks(ticks)
            
        # Case 2: Categorical - Set first and last thicks bounds as index
        else:
            ticks = tick_positions_y if side == 'left' else tick_positions_x
            low_b = 0
            up_b  = len(ticks)-1
        
        ax.spines[side].set_bounds(low_b, up_b)
        
        
def get_colors(n_cols: int, colormap: str = 'jet') -> NDArray:
    """
    Generates a list of colors by sampling evenly from a specified colormap.

    :param n_cols: The number of colors to generate.
    :type n_cols: int
    :param colormap: The name of the colormap to sample from, defaults to 'jet'.
    :type colormap: str, optional
    :return: An array of RGBA color values.
    :rtype: NDArray
    """
    cmap = plt.cm.get_cmap(colormap) # Get the colormap
    # Generate n_cols evenly spaced values between 0 and 1
    # These will be used to sample from the colormap
    values = np.linspace(0, 1, n_cols)
    # Get the colors from the colormap
    colors = cmap(values)

    return colors

def get_color_from_name(color_name: str, colormap: str ='jet') -> Union[NDArray, None]:
    """
    Returns the closest color from a specified colormap to a given named color.

    :param color_name: The name of the color to match.
    :type color_name: str
    :param colormap: The name of the colormap to sample from, defaults to 'jet'.
    :type colormap: str, optional
    :return: An RGBA color value from the colormap that is closest to the named color, or None if the color name is not recognized.
    :rtype: np.ndarray | None
    """
    cmap = plt.cm.get_cmap(colormap)  # Get the colormap
    colors = cmap(np.linspace(0, 1, 256))  # Get 256 colors from the colormap
    named_colors = matplotlib.colors.cnames  # Get named colors

    if color_name in named_colors:
        color_rgb = matplotlib.colors.to_rgb(color_name)  # Convert named color to RGB
        color_rgb = list(color_rgb)+[1]  # Add alpha channel
        # Find the closest color in the colormap
        color_diffs = colors - color_rgb
        color_diff_magnitudes = np.linalg.norm(color_diffs, axis=1)
        closest_color_index = np.argmin(color_diff_magnitudes)
        closest_color = colors[closest_color_index]
        return closest_color
    else:
        print(f"{color_name} is not a recognized color name.")
        return None