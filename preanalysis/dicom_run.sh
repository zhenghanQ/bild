#!/bin/bash
python /om/project/bild/Analysis/task_openfmri/scripts/preanalysis/dicomconvert2.py -d /mindhive/xnat/dicom_storage/BILD/Kids/%s/dicom/*.dcm  -s BILDC3348 -c dcm2nii -o /om/project/bild/Analysis/task_openfmri/openfmri -f /om/project/bild/Analysis/task_openfmri/scripts/preanalysis/heuristic.py
