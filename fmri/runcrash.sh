#!/bin/bash
pars=(BILDC3255)

for i in "${pars[@]}"

do 

    mkdir -p /om/project/bild/Analysis/task_openfmri/scripts/fmri/${i}
    echo '#!/bin/bash' > /om/project/bild/Analysis/task_openfmri/scripts/fmri/${i}/l1_openfmri.sh
    echo '#SBATCH --mem=10G' >> /om/project/bild/Analysis/task_openfmri/scripts/fmri/${i}/l1_openfmri.sh
    echo '#SBATCH -c2' >> /om/project/bild/Analysis/task_openfmri/scripts/fmri/${i}/l1_openfmri.sh
    echo '#SBATCH --time=2-23:00:00' >> /om/project/bild/Analysis/task_openfmri/scripts/fmri/${i}/l1_openfmri.sh
    echo "i=$i" >> /om/project/bild/Analysis/task_openfmri/scripts/fmri/${i}/l1_openfmri.sh
    echo "python /om/project/bild/Analysis/task_openfmri/scripts/fmri/fmri_ants_openfmri_sparse_3mm.py --sd /mindhive/xnat/surfaces/BILD/Kids --target /om/user/ysa/OASIS-30_Atropos_template_in_MNI152_2mm.nii.gz -d /mindhive/xnat/data/BILD/nifti_data -t 1 -m 1 -x 'BILDC*' -s ${i} -w /om/project/bild/Analysis/task_openfmri/first_level/crashed_working -o /om/project/bild/Analysis/task_openfmri/first_level/crashed_output_dir -p 'SLURM'" >> /om/project/bild/Analysis/task_openfmri/scripts/fmri/${i}/l1_openfmri.sh

    #cat /om/project/bild/Analysis/task_openfmri/scripts/fmri/${i}/l1_openfmri.sh
    
    cd /om/project/bild/Analysis/task_openfmri/scripts/fmri/${i}/

    sbatch l1_openfmri.sh

done  
