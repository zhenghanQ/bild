import os
import numpy as np
import pandas as pd

# so that the participant module can be loaded
script_dir = "/om/project/bild/Analysis/task_openfmri/scripts/fmri"
behav_file = "/om/project/bild/Analysis/task_openfmri/scripts/groups/behav.txt"
cur_dir = os.getcwd()

if cur_dir != script_dir:
    os.chdir(script_dir)

# custom python file made by me
import participant as pa

read_pars_from_path = False
base_path = "/om/project/bild/Analysis/task_openfmri/first_level/output_dir_071616/model01/task001"
make_file=True

if make_file:
    df = pd.read_csv(behav_file, sep='\t')

use_num = "01"
use_type = "copes"
use_mni = True

# get the required information for each participant
par_objs = {}
for (idx, par) in enumerate(df["subject_id"]):
    # hopefully this will take care of the SLI
    if "SLI" in par:
        par = par.split("SLI")
        par = par[0] + "A" + par[-1]
        print(par)
        df.iloc[idx, 0] = par
    try:
        par_objs[par] = pa.Participant(base_path, par, use_num, use_type, use_mni)
    except:
        par_objs[par] = "ERROR"

missing = [key for key, val in par_objs.items() if val == "ERROR"]

if missing: 
    if len(missing) == 1:
        drop_idx = np.where(df["subject_id"] == missing[0])[0][0]
        df = df.drop(df.index[drop_idx]).reset_index(drop=True)
    else:
        drop_idx = np.where(df["subject_id"] == missing[0])[0]
        df = df.drop(df.index[drop_idx]).reset_index(drop=True)

all_files_to_merge = ""
for par in par_objs.values():
    if not (par == "ERROR"):
        all_files_to_merge += (" " + par.afile)

design_file_cat = df[["young_kid", "old_kid", "adult", "composite_motion"]]
design_file_cont = df[["age", "composite_motion"]]
design_file_cat.to_csv("des_cat.txt", index=False, header=False, sep=' ')
design_file_cont.to_csv("des_cont.txt", index=False, header=False, sep=' ')

contrast = [1, 1, 1]

cmd = "fslmerge -t merged_cope1_76par_mni_4D.nii.gz {}".format(all_files_to_merge.strip())
out = pa.shell(cmd.split(), split=True)


# checking motion
f1 = "/om/project/bild/Analysis/task_openfmri/scripts/motion/motionfiles/lessthan20%motiontask1.csv"
