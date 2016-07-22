#!/bin/bash
for file in /om/project/bild/Analysis/task_openfmri/openfmri/BILDC*; do
    basefile=${file##*/}
    if ! [[ -d "/mindhive/xnat/surfaces/BILD/Kids/$basefile" ]]; then
        echo "$basefile surface not generated"
    fi
done
