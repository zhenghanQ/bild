import os
import subprocess
import utils

aparc_file = "/om/project/bild/Analysis/task_openfmri/misc/aparc.a2009s+aseg_2mm_MNI.nii.gz"
color_lut_txt = "/cm/shared/openmind/freesurfer/6.0.0/FreeSurferColorLUT.txt"
output_dir = "/om/project/bild/Analysis/task_openfmri/misc/freesurfer_aparc_aseg_masks_mni"

cortical_aparc_aseg_2009_idxs = [
    range(1127, 1203),
    range(1204, 1280),
]

subcortical_aparc_aseg_2009_idxs = [
    range(4, 79),
    range(80, 90),
    range(91, 94),
    [96, 97, 98]
]

def fsl_cmd(thresh, aparc_aseg, out_name):
    cmd = ("fslmaths {aparc_aseg} -uthr {thresh1} -thr {thresh2} "
           "{out_name}").format(
                aparc_aseg=aparc_aseg, 
                thresh1=thresh,
                thresh2=thresh,
                out_name=out_name
            )
    return cmd.split()

def extract_freesurfer_lut_info(color_lut_path, label_ranges):
    """color_lut_path: string - path, slidx: int - start line number,
    elidx: int- end line number
    """
    with open(color_lut_path) as buf:
        cl_info = buf.readlines()
    preproccessed_info = []
    for sub_range in label_ranges:
        for idx in sub_range:
            line_info = cl_info[idx].strip().split()
            try:
                preproccessed_info.append([idx, int(line_info[0]), line_info[1]])
            except:
                preproccessed_info.append([idx, "nothing", "nothing"])
    return preproccessed_info


masks_info_cortical = extract_freesurfer_lut_info(color_lut_txt, cortical_aparc_aseg_2009_idxs)
masks_info_subcortical = extract_freesurfer_lut_info(color_lut_txt, subcortical_aparc_aseg_2009_idxs) 

for masks_info in [masks_info_cortical, masks_info_subcortical]:
    for idx in range(len(masks_info)):
        if not (masks_info[idx][1] == "nothing"):
            try:
                save_path = os.path.join(output_dir, masks_info[idx][2])
                utils.shell(fsl_cmd(masks_info[idx][1], aparc_file, save_path))
            except:
                print(masks_info[idx], "didn't work")
