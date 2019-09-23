import os
import utils
import numpy as np
import pandas as pd

def load_data_and_preproc(info_dict, vox_thresh=27):
    """info_dict: dictionary with the following key - value pairs:
    base_path: path for all the required files
    voxels: path for the tabular file with the number of voxels per ROI
    behav: path for the tabular file with the behavior data per subject
    data: path for the tabular file with the roi data
    returns: multiple pd.DataFrames
    """
    voxels = pd.read_csv(os.path.join(info_dict["base_path"], info_dict["voxels"]), index_col=0)
    behav = pd.read_csv(os.path.join(info_dict["base_path"], info_dict["behav"]), sep="\t", index_col=0)
    data = pd.read_csv(os.path.join(info_dict["base_path"], info_dict["data"]), index_col=0)

    vx27 = voxels.iloc[:, (voxels.iloc[1, :] > vox_thresh).values]
    rm_fidx = []

    cols_to_remove = [
        "Vent", "Stem", "Cerebellum", "CSF", "White", "plexus"
    ]

    for idx, col in enumerate(vx27.columns):
        for f in cols_to_remove:
            if f in col:
                rm_fidx.append(idx)

    rm_f_idx = np.unique(rm_fidx)
    vx_data_thresh = vx27.iloc[:, np.setdiff1d(np.arange(vx27.shape[1]), rm_f_idx)]
    roi_data = data.loc[:, vx_data_thresh.columns]

    print(np.all(roi_data.columns == vx_data_thresh.columns))

    data_sets = {}
    proc_roi_data = utils.projection(roi_data, behav.loc[:, ["gender", "iq", "composite_motion"]])
    
    yk_ad_idx = np.logical_or(behav.young_kid == 1, behav.adult == 1)
    yk_ok_idx = np.logical_or(behav.young_kid == 1, behav.old_kid == 1)
    ok_ad_idx = np.logical_or(behav.old_kid == 1, behav.adult == 1)
    
    data_sets["roi_data"] = roi_data
    data_sets["behav"] = behav
    data_sets["proc_roi_data"] = proc_roi_data
    data_sets["kids_adults"] = proc_roi_data[yk_ad_idx]
    data_sets["kids_adults_behav"] = behav[yk_ad_idx]
    data_sets["kids_adoles"] = proc_roi_data[yk_ok_idx]
    data_sets["kids_adoles_behav"] = behav[yk_ok_idx]
    data_sets["adols_adults"] = proc_roi_data[ok_ad_idx]
    data_sets["adols_adults_behav"] = behav[ok_ad_idx]

    return data_sets


