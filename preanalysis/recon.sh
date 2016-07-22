#!/bin/bash

subjs=(BILDC3345 BILDC3346 BILDC3271 BILDC3327 BILDC3348 BILDC3276 BILDC3299 BILDC3290 BILDCD001 BILDCD000 BILDC3105 BILDC3125 BILDC3151 BILDC3152 BILDC3074 BILDC3161 BILDC3213 BILDC3093 BILDC3092 BILDC3095 BILDC3268 BILDC3224 BILDC3139 BILDC3141 BILDC3138 BILDC3223 BILDC3083 BILDC3267 BILDA3053 BILDC3220)

data_dir=/om/project/bild/Analysis/task_openfmri/openfmri

for sub in "${subjs[@]}";do
    if [ -d /mindhive/xnat/surfaces/BILD/Kids/${sub} ]; then
        echo '#!/bin/bash' > /om/project/bild/Analysis/task_openfmri/scripts/temp/${sub}_recon.sh
        echo '#SBATCH --mem=8G' >> /om/project/bild/Analysis/task_openfmri/scripts/temp/${sub}_recon.sh
        echo "recon-all -s ${sub} -all -hippo-subfields" >> /om/project/bild/Analysis/task_openfmri/scripts/temp/${sub}_recon.sh
    fi
    if [ ! -d /mindhive/xnat/surfaces/BILD/Kids/${sub} ]; then
        echo '#!/bin/bash' > /om/project/bild/Analysis/task_openfmri/scripts/temp/${sub}_recon.sh
	echo '#SBATCH --mem=12G' >> /om/project/bild/Analysis/task_openfmri/scripts/temp/${sub}_recon.sh
	echo '#SBATCH --workdir=/om/project/bild/Analysis/task_openfmri/scripts/temp'
	echo "recon-all -s ${sub} -all -hippo-subfields -i ${data_dir}/${sub}/anatomy/T1_001.nii.gz -sd /mindhive/xnat/surfaces/BILD/Kids/" >> /om/project/bild/Analysis/task_openfmri/scripts/temp/${sub}_recon.sh
    fi
    sbatch /om/project/bild/Analysis/task_openfmri/scripts/temp/${sub}_recon.sh
done
