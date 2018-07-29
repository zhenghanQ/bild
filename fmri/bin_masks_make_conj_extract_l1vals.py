import os
from nilearn.image import math_img
from nilearn.masking import apply_mask
import utils
import pandas as pd
import numpy as np

def bin_conj(work_dir, zstat_path, zstat_thresh, roi_path):
    """work_dir: string - path, zstat_path: string - path,
    zstat_thresh: float - zthreshold, roi_path: string - path
    """
    roi_name = roi_path.split("/")[-1].split(".")[0]
    if os.getcwd() != work_dir:
        os.chdir(work_dir)
    temp_name = "{}_zstat_roi".format(roi_name)
    if not os.path.isdir(temp_name):
        os.makedirs(temp_name)
    os.chdir(temp_name)
    new_work_dir = os.getcwd()
    roi_bin = math_img("img > .5", img=roi_path)
    zstat_bin = math_img("img > {}".format(zstat_thresh), img=zstat_path)
    roi_bin.to_filename("roi_bin.nii.gz")
    zstat_bin.to_filename("zstat_bin.nii.gz")
    roi_bin_path = os.path.join(new_work_dir, "roi_bin.nii.gz")
    zstat_bin_path = os.path.join(new_work_dir, "zstat_bin.nii.gz")
    conj_bin = math_img("img1*img2", img1=roi_bin_path, img2=zstat_bin_path)
    final_save_name = "{}_zstat_conj_bin.nii.gz".format(roi_name)
    conj_bin.to_filename(final_save_name)
    return new_work_dir, os.path.join(new_work_dir, final_save_name)

def extract_from_cope(cope_path, mask_path):
    """cope_path: string - path, mask_path: string - path
    """
    vals = apply_mask(cope_path, mask_path)
    if vals.shape[0] == 1:
        return vals.ravel()
    return vals

# run remaining preprocessing steps     
zthresh = 3.09
cope_dir = "/om/project/bild/Analysis/task_openfmri/first_level/output_dir_071616/model01/task001/{}/copes/mni/cope02.nii.gz"
zstat_path = "/om/project/bild/Analysis/task_openfmri/second_level/l2output_29_f1_sd_3.09/output_dir/model001/task001/all/stats/contrast_2/zstat1_threshold.nii.gz"
work_dir = "/om/scratch/Sat/ysa/"
roi_path = "/om/project/bild/Analysis/task_openfmri/misc/freesurfer_aparc_aseg_masks_mni/"
sub_dict = utils.read_pickle("/om/project/bild/Analysis/task_openfmri/scripts/fmri/subj_dict_76.pkl")
rois = os.listdir(roi_path)
allsubs = [i for x in sub_dict.keys() for i in sub_dict[x]]

df_vals = pd.DataFrame(
    data=np.zeros((len(allsubs), len(rois))),
    index=allsubs,
    columns=rois
)

df_voxels = df_vals.copy()

for roi in rois:
    this_roi_path = os.path.join(roi_path, roi)
    _, mp = bin_conj(work_dir, zstat_path, zthresh, this_roi_path)
    for sub in allsubs:
        try:
            vals = extract_from_cope(cope_dir.format(sub), mp)
            df_voxels.loc[sub, roi] = len(vals)
            df_vals.loc[sub, roi] = vals.mean()        
        except:
            pass

# remove all zero columns
cols_nonzero = np.where(df_voxels.mean(0) != 0.0)[0]
df_voxels_nz = df_voxels.iloc[:, cols_nonzero]
df_vals_nz = df_vals.iloc[:, cols_nonzero]
diff_n_voxels = np.where(df_voxels_nz.std(0) != 0.0)

save_path = "/om/project/bild/Analysis/task_openfmri/second_level/fszstat_roiconj"
df_vals.to_csv(os.path.join(save_path, "fszstatcope2_means_all.csv"))
df_voxels.to_csv(os.path.join(save_path, "fszstatcope2_nvoxels_all.csv"))
df_voxels_nz.to_csv(os.path.join(save_path, "fszstatcope2_nvoxels_nz.csv"))
df_vals_nz.to_csv(os.path.join(save_path, "fszstatcope2_means_nz.csv"))
