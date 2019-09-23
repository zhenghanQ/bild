#!/bin/bash
#SBATCH --mem=10G
#SBATCH -c2
#SBATCH --time=6-23:00:00
for task in 1 2; do python /om/project/bild/Analysis/task_openfmri/scripts/fmri/fmri_ants_openfmri_sparse_3mm.py --sd /mindhive/xnat/surfaces/BILD/Kids --target /om/project/bild/Analysis/task_openfmri/misc/OASIS-30_Atropos_template_in_MNI152_2mm.nii.gz -d /om/project/bild/Analysis/task_openfmri/openfmri -t $task -m 1 -x "BILD*" -w /om/project/bild/Analysis/task_openfmri/first_level/working_dir_071616/BILDC3165 -o /om/project/bild/Analysis/task_openfmri/first_level/output_dir_071616 -p 'SLURM' --plugin_args "dict(sbatch_args='-N1 -c2 --mem=10G', max_jobs=50)" ; done
