#!/bin/bash
for file in /mindhive/xnat/dicom_storage/BILD/Kids/*; do
    basefile=${file##*/}
    if ! [[ -d "/om/project/bild/Analysis/task_openfmri/openfmri/$basefile" ]]; then
        echo "$basefile is not present in target location."
    fi
done
