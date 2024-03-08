import numpy as np
from pandas import DataFrame

def get_p_ors(ori: int|str) -> list[int]:
    """Given an orientation, returns a list of parallel orientations (e.g. 0° -> 0° and 180°).

    :param ori: The input orientation.
    :type ori: int | str
    :return:  A list containing the input orientation and its parallel orientations.
    :rtype: list[int]
    """

    ori = int(ori); p_ori = [ori]; flat_ors = [0,180,360]

    if ori in flat_ors:
        p_ori = flat_ors
    else:
        #the else below is a elif ori>180 (the case where ori = 180 is
        # handled in the case above): 
        p_ori.append(ori+180) if ori<180 else p_ori.append(ori-180)
    return p_ori

def get_ortho_ors(ori: int| str) -> list[int]:
    """Given an orientation, returns a list of orthogonal orientations (e.g. 0° -> 90° and 270°).

    :param ori: The input orientation.
    :type ori: int | str
    :return: A list containing the orthogonal orientations corresponding to the parallel orientations.
    :rtype: list[int]
    """

    ori = int(ori); ortho_ors = []
    p_ori = get_p_ors(ori)
    for p_or in p_ori:
        ortho_ors.append(p_or+90) if p_or<270 else ortho_ors.append(p_or-360+90)
    return ortho_ors

def compute_OSI(tuningC_avg: dict)-> DataFrame:
    #TODO: controlla, ci sono problemi solo con i ori flat (0,180)
    """ Compute Orientation Selectivity Index (OSI) from 
    the average tuning curve tuningC_avg.

    :param tuningC_avg: Dictionary containing orientation tuning 
                        curve means for each cell.
    :type tuningC_avg: dict
    :return: DataFrame containing the OSI values.
    :rtype: DataFrame
    """

    tuningC_df = DataFrame(tuningC_avg)
    best_ori = tuningC_df.idxmax(axis=1).to_numpy()
    OSI_v = np.full_like(best_ori, np.nan)

    for r_idx, b_or in enumerate(best_ori):
        p_ors = get_p_ors(b_or)
        if '360' not in tuningC_avg.keys() and 360 in p_ors:
            p_ors.remove(360)
        ortho_ors = get_ortho_ors(b_or)
        
        R_pref = tuningC_df.loc[r_idx,[b_or]].to_numpy()
        R_ortho = np.nanmean(tuningC_df.loc[r_idx,
                    [str(ori) for ori in ortho_ors]])
        OSI_v[r_idx] = (R_pref -R_ortho)/(R_pref + R_ortho)

    tuningC_df['Preferred or'] = best_ori
    tuningC_df['OSI'] = OSI_v; 
    tuningC_df['OSI'] = tuningC_df['OSI'].astype(float)

    return tuningC_df