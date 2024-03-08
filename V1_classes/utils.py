
import glob
from itertools import combinations
import json
import os.path as path
import re
import shutil
from typing import Any, Union
import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame
from torch import Tensor
from functools import reduce


def read_json(path: str) -> dict[str, Any]:
    '''
    Read JSON data from a file.

    :param path: The path to the JSON file.
    :type path: str
    :raises FileNotFoundError: If the specified file is not found.
    :return: The JSON data read from the file.
    :rtype: Dict[str, Any]
    '''
    
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f'File not found at path: {path}')
    
    
def remove_dirs(fp: str, dirs2rm: Union[str, list[str]]) -> None:
    """
    Remove specified directories in the given filepath fp.

    :param fp: the filepath where directories of interest are present
    :type fp: str
    :param dirs2rm: A single folder name or a list of folder names to be removed.
    :type dirs2rm: Union[str, list[str]]
    """
    
    dirs2rm = [dirs2rm] if isinstance(dirs2rm, str) else dirs2rm

    for f in dirs2rm:
        if path.isdir(path.join(fp, f)):
            shutil.rmtree(path.join(fp,f))
            
def find_files_by_extension(dir: str, extension: str, recursive: bool = False) -> list:
    """
    Find all files with a certain extension inside a directory (including dirs inside of it)
    
    :param dir: the main directory where to search for files
    :type dir: str
    :param extension: the extension of the files to search (e.g. .txt, .xlsx,...)
    :type extension: str
    :param recursive: decides if to extend the search to directories inside
    :type recursive: bool
    :return: list of filepaths corresponding to the files found
    :rtype: list
    """
    pattern = path.join(dir, f"**/*{extension}") if recursive else path.join(dir, f"*{extension}")

    return glob.glob(pattern, recursive=recursive)


def contains_character(s: str, pattern: str | list[str]) -> bool | list[bool]:
    """
    Check if the given string contains at least one character that matches any of the specified patterns.

    :param s: The input string to check
    :type s: str
    :param pattern: The pattern or list of characters to look for.
    :type pattern: str | list[str]
    :return: True if the string contains at least one character matching any of the specified patterns, False otherwise.
    :rtype: bool | list[bool]
    """

    if isinstance(pattern, list):
        # If pattern is a list, join the characters with '|' for the regex
        pattern = '|'.join(re.escape(char) for char in pattern)

    return bool(re.search(pattern, s))


def delete_fake_cells(phys_rec: NDArray) -> NDArray:
    """
    Deletes rows (cells) in a matrix that contain only a single repeated value.
    
    :param phys_rec: The input matrix (physical recording).
    :type phys_rec: NDArray
    :return: The matrix with rows containing a single repeated value removed.
    :rtype: NDArray
    """
    # Find the indices of rows with a single repeated value (e.g. 00000000000)
    index_of_repeated_row = np.where((phys_rec == phys_rec[:, :1]).all(axis=1))[0]
    if len(index_of_repeated_row) > 0:
        print(f"The rows at indices {index_of_repeated_row} are constituted only by a single value repeated.")
        phys_rec = np.delete(phys_rec, index_of_repeated_row, axis=0) # Remove the identified rows from the matrix
    return phys_rec


def load_2p_data(Analysis_settings:dict[str,Any]) -> NDArray:
    """
    load 2 photon imaging data and do essential preprocessing

    :param Analysis_settings: dictionary containing analysis-related global params
    :type Analysis_settings: dict[str,Any]
    :return: matrix of preprocessed 2p data
    :rtype: NDArray
    """
    F_ds = {k.split('.')[0]: np.load(k,allow_pickle=True) for k in Analysis_settings['phys_fn']}
    F_ds['F_minus_neu'] = F_ds['F'] - 0.7*F_ds['Fneu']
    F_ds['F_minus_neu'][F_ds['F_minus_neu']<0]=0 
    iscell = F_ds['iscell'][:,0]==1
    for k,v in F_ds.items():
        if 'F' in k:
           F_ds[k] = delete_fake_cells(v[iscell])
    return F_ds



def Stim_var_rename(stim_df: DataFrame, stim_var:str) -> DataFrame:
    """
    rename grays with gray+previous orientation.

    :param stim_var: variable indicating stimulus conditions in your stim_df
    :type stim_var: str
    :param stim_df: dataframe containing stimuli presented and their timings
    :type stim_df: DataFrame
    :return: stimulus dataframe with renamed stimuli
    :rtype: DataFrame
    """
    
    #i add to every gray name the preceding orientation
    renamed_st = []
    for st in stim_df[stim_var].tolist():
        curr_st = str(st); last_ch = curr_st[-1]
        num = curr_st.split('.')[0]
        if contains_character(curr_st, pattern = r'\d+'):
            renamed_st.append(num+last_ch if last_ch in ['+','-'] else curr_st)
        elif curr_st=='gray':
            #rename current gray adding the ori of the precedent stim
            ori = renamed_st[-1]
            renamed_st.append('gray '+ str(ori))
        else:
            renamed_st.append(curr_st)
    
    stim_df[stim_var] = renamed_st
    return stim_df




def add_keys_logicalDict(logical_dict: dict[str, NDArray]) -> dict[str, NDArray]:
    """
    Add integrated keys to the logical dictionary (e.g. 180, +, gray -, ...).

    :param logical_dict: The input logical dictionary.
    :type logical_dict: dict[str, NDArray]
    :return: The logical dictionary with added keys.
    :rtype: dict[str, NDArray]
    """
   
    stim_names = logical_dict.keys()
    #check if any element of stim_names contains a '+' sign
    if any(contains_character(string, pattern=r'\+') for string in stim_names): 
        #ora creo le voci integrate: orientamenti, direzioni drift [+,-] e relativi grays
        pattern = r'\d+\.\d+|\d+'  #pattern to find ints or floats in a string
        #get all numeric keys (e.g. 180, 90,...)
        new_keys = [re.findall(pattern, elem)[0] for elem in stim_names if re.findall(pattern, elem)]
        #integrate general orientations (+,-) and intertrials
        new_keys = list(set(new_keys))+['+','-']; new_keys =new_keys + ['gray '+ n for n in new_keys]
        
        for new_key in new_keys:
            plus_minus = new_key[-1] #this is need just for non num keys
            if "+" not in new_key and "-" not in new_key: #i.e. if new_key is a number
                key_plus = new_key+'+'; key_minus = new_key+'-'; alt_keys = [key_plus, key_minus]
            elif 'gray' in new_key: #only gray+  and gray- should enter here
                alt_keys = [key for key in logical_dict.keys() if 'gray' in key and plus_minus in key]
            else:
                alt_keys = [key for key in logical_dict.keys() if not('gray' in key) and plus_minus in key]
            # Concatenate the arrays vertically (axis=0) to form a single array
            arrays_list = []
            for k in alt_keys:
                try:
                    arrays_list.append(logical_dict[k])
                except:
                    print(k+' is missing')
            concatenated_array = np.concatenate(arrays_list, axis=0)
            # Sort the rows in ascending order based on the first column (index 0)
            logical_dict[new_key] = concatenated_array[np.argsort(concatenated_array[:, 0])] 
    return logical_dict

def convert_to_numpy(data: Union[list, tuple, Tensor, DataFrame]) -> NDArray:
    """
    convert data of different formats into numpy NDArray
    
    :param data: your dataset. It will be converted into a numpy ndarray
    :type data: Union[list, tuple, NDArray, Tensor, DataFrame]
    
    :return: data converted into  numpy ndarray
    :rtype: NDArray
    """
    try:
        if isinstance(data, DataFrame):
            numpy_array = data.to_numpy()
        elif isinstance(data, Tensor):
            numpy_array = data.numpy()
        else:
            numpy_array = np.array(data)
        return numpy_array
    except Exception as e:
        print(f"Error during numpy array conversion: {e}")
        
def SEMf(data: Union[list[float], tuple[float], NDArray], 
         axis: int = 0) -> NDArray:
    """
    Compute standard error of the mean (SEM) for a sequence of numbers

    :param data: your dataset. It will be converted into a numpy ndarray
    :type data: Union[List[float], Tuple[float], NDArray]
    :param axis: axis used to compute SEM (valid for NDArray data only)
    :type axis: int
    :return: The computed SEMs
    :rtype: NDArray
    """
    try:
        #  convert data into NDArray
        data_array = convert_to_numpy(data)
        
        # compute standard deviation and sample size on the specified axis
        std_dev = np.nanstd(data_array, axis=axis) if data_array.ndim > 1 else np.nanstd(data_array)
        sample_size = data_array.shape[axis]
        
        # compute SEM
        sem = std_dev / np.sqrt(sample_size)
        return sem
    
    except Exception as e:
        print(f"Error during SEM computation: {e}")
 
def get_idxs_above_threshold(measures: NDArray, threshold: float, 
                             dict_fields: list[str] = ['cell_nr', 'idxs_above_threshold', 
                            'nr_above_threshold', 'fraction_above_threshold']) -> dict[str, Any]:
    """Retrieves indexes and frequency of elements above a certain threshold in an array.

    :param measures: NumPy array containing the measurements.
    :type measures: NDArray
    :param threshold: Threshold for considering a measurement above the threshold.
    :type threshold: float
    :param dict_fields: List of fields for the output dictionary
        defaults to ['cell_nr', 'idxs_above_threshold', 'nr_above_threshold', 'fraction_above_threshold']
    :type dict_fields: list[str], optional
    :return: Dictionary containing indexes and frequency of elements above threshold.
    :rtype: dict[str, Any]
    """

    above_thr = {}
    # Calculate the number of samples
    sample_sz = measures.shape[0]; above_thr[dict_fields[0]] = sample_sz 
    # Find the indices above the threshold 
    idxs_above_thr = np.where(measures>threshold)[0]; above_thr[dict_fields[1]] = idxs_above_thr
    #absolute and relative frequency of indices above threshold 
    nr_above_thr = len(idxs_above_thr); rf_above_thr = nr_above_thr/sample_sz
    #add frequency info into the dictionary
    above_thr[dict_fields[2]] = nr_above_thr; above_thr[dict_fields[3]] = rf_above_thr
    return above_thr

def get_relevant_cell_stats(cell_stats_df: DataFrame, 
                            thresholds_dict: dict[str, float]) -> dict[Union[str, tuple[str, ...]], dict[str, Union[NDArray, int, float]]]:
    """Retrieves statistical information for relevant cell statistics based on given thresholds.

    :param cell_stats_df:  DataFrame containing cell statistics.
    :type cell_stats_df: pd.DataFrame
    :param thresholds_dict: Dictionary mapping statistic names to their corresponding thresholds.
    :type thresholds_dict: Dict[str, float]
    :return: Dictionary containing the summary statistics.
    :rtype: Dict[Union[str, Tuple[str, ...]], Dict[str, Union[NDArray, int, float]]]
    """
    cell_stats_df = cell_stats_df.loc[:, list(thresholds_dict.keys())] 
    all_key_combinations = [comb for r in range(1, len(cell_stats_df) + 1) for comb in combinations(cell_stats_df.keys(), r)]
    stats_dict = {}
    for combination in all_key_combinations:
        if len(combination)==1:
            stats_dict[combination[0]] = get_idxs_above_threshold(cell_stats_df[combination[0]], thresholds_dict[combination[0]])
        else: # Combination of multiple metrics
            metrics = " and ".join(combination); stats_dict[metrics]={}
            # Calculate intersection of indices above threshold for multiple metrics
            stats_dict[metrics]['idxs_above_threshold'] = reduce(np.intersect1d, ([stats_dict[m]['idxs_above_threshold'] for m in combination]))  
            stats_dict[metrics]['nr_above_threshold'] = len(stats_dict[metrics]['idxs_above_threshold']) #absolute frequency
            #relative frequency
            stats_dict[metrics]['fraction_above_threshold'] = stats_dict[metrics]['nr_above_threshold']/stats_dict[combination[0]]['cell_nr'] 
    return stats_dict


def multioption_prompt(opt_list: list[str], 
                       in_prompt: str) -> Union[str, list[str]]:
    """
    Prompt the user to choose from a list of options.
    
    :param opt_list: list of options
    :type opt_list: list[str]    
    :param in_prompt: Prompt message to display.
    :type in_prompt: str
    :return: Either a single option (str) or a list of options (List[str])
            selected
    :rtype: Union[str, list[str]]
    """
    # Generate option list
    opt_prompt = '\n'.join([f'{i}: {opt}' for i, opt in enumerate(opt_list)])
    # Prompt user and evaluate input
    idx_answer = eval(input(f"{in_prompt}\n{opt_prompt}"))
    
    # Check if the answer is a list
    if isinstance(idx_answer, list):
        answer = [opt_list[idx] for idx in idx_answer]
    else: # If not a list, return the corresponding option
        answer = opt_list[idx_answer]

    return answer


def create_variable_dict(locals_or_globals: dict, 
                         variables_list: list) -> dict:
    """
    Create a dictionary associating each variable name to the variable 
    in the local or global namespace.
    
    :param locals_or_globals: Dictionary-like object representing local 
                            or global namespace.
    :type locals_or_globals: dict
    :param variables_list: list of variable names of interest
    :type variables_list: list
    :return: Dictionary where keys are variable names and 
            values are the corresponding variables.
    :rtype: dict
    """
    variable_dict = {}
    for var in variables_list:
        if var in locals_or_globals:
            variable_dict[var] = locals_or_globals[var]
        else:
            print('\033[1mVariable {} not found\033[0m'.format(var))

    return variable_dict