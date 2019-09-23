import os
from nilearn.image import resample_to_img
from niworkflows.interfaces.mni import RobustMNINormalization
from nipype.interfaces.ants import ApplyTransforms
import utils

def make_nii_gz(file_dir, write_dir):
    file_to_cp = os.path.join(file_dir, "brain.mgz")
    cp_path = os.path.join(write_dir, "brain_copy.mgz")
    print(utils.shell("cp {} {}".format(file_to_cp, cp_path).split()))
    os.chdir(write_dir)
    cmd = "mri_convert {brain_copy_mgz} {brain_nii}".format(
        brain_copy_mgz = "brain_copy.mgz",
        brain_nii = "brain.nii.gz"
    )
    print(utils.shell(cmd.split()))
    print(utils.shell("rm brain_copy.mgz".split()))

def transform_data(subject_surf_dir, mni_template, roi_name, write_dir):
    subj = subject_surf_dir.split("/")[-1]
    if not os.path.isdir(subject_surf_dir):
        raise Exception("provide full path to subject dir")
    #work_dir = os.path.join(subject_surf_dir, "mri")
    work_dir = os.path.join(write_dir, subj)
    if not os.path.isdir(work_dir):
        os.makedirs(work_dir)
    os.chdir(work_dir)
    subj_dir_mri = os.path.join(subject_surf_dir, "mri")
    brain_mgz = os.path.join(subj_dir_mri, "brain.mgz")
    if not os.path.isfile(brain_mgz):
        raise Exception("subject does not have brain.mgz")
    roi_file_path = os.path.join(subj_dir_mri, roi_name)
    if not os.path.isfile(roi_file_path):
        raise Exception("subject does not have {} roi".format(roi_name))
    make_nii_gz(subj_dir_mri, work_dir)
    roi_resampled = resample_to_img(roi_file_path, os.path.join(work_dir, "brain.nii.gz"))
    resampled_roi_path = os.path.join(work_dir, roi_name.split(".")[0] + "_resampled.nii.gz")
    roi_resampled.to_filename(resampled_roi_path)
    norm = RobustMNINormalization()
    norm.inputs.moving_image = os.path.join(work_dir, "brain.nii.gz")
    norm.inputs.template = "mni_icbm152_linear"
    norm.inputs.template_resolution = 2
    print("running normalization for:\n{}".format(subject_surf_dir))
    res = norm.run()
    applyt = ApplyTransforms(dimension=3, interpolation="NearestNeighbor")
    applyt.inputs.input_image = resampled_roi_path
    applyt.inputs.reference_image = mni_template
    applyt.inputs.transforms = res.outputs.composite_transform
    print("running apply transform for:\n{}".format(subject_surf_dir))
    applyt.run()

# single run test
#transform_data(os.path.join(surf_dir, "Kids", sub), fsl_mni_brain_2mm, "leftHippo_bin.nii.gz", write_dir)

# run the script
surf_dir = "/mindhive/xnat/surfaces/BILD/"
fsl_mni_brain_2mm = "/cm/shared/openmind/fsl/5.0.9/data/standard/MNI152_T1_2mm_brain.nii.gz"
subj_dict = utils.read_pickle("/om/project/bild/Analysis/task_openfmri/scripts/fmri/subj_dict_76.pkl")      
write_dir = "/om/scratch/Wed/ysa"

for key in subj_dict.keys():
    surf_path = os.path.join(surf_dir, key)
    for sub in subj_dict[key]:
        print("trying {}".format(sub))
        try:
            # stuff
            transform_data(
                os.path.join(surf_path, sub), 
                fsl_mni_brain_2mm, 
                "leftHippo_bin.nii.gz", 
                write_dir
            )
        except:
            print("sub {} did not run".format(sub))
