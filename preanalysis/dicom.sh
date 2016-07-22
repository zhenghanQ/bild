#!/bin/bash
mkdir temp
temp_dir=/om/project/bild/Analysis/task_openfmri/scripts/temp
subjs=(BILDC3327 BILDC3345 BILDC3346 BILDC3348)
for sub in "${subjs[@]}"
do
mkdir ${temp_dir}/${sub}_dicom
cd ${temp_dir}/${sub}_dicom
echo '#!/bin/bash' > dicom_run.sh
echo "python /om/project/bild/Analysis/task_openfmri/scripts/preanalysis/dicomconvert2.py -d /mindhive/xnat/dicom_storage/BILD/Kids/%s/dicom/*.dcm  -s ${sub} -c dcm2nii -o /om/project/bild/Analysis/task_openfmri/openfmri -f /om/project/bild/Analysis/task_openfmri/scripts/preanalysis/heuristic.py" >> dicom_run.sh
done
for sub in "${subjs[@]}"
do
cd ${temp_dir}/${sub}_dicom
sbatch --mem=8GB dicom_run.sh
done

