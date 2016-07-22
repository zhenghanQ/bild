import os
import subprocess
import numpy as np
import shutil


if __name__ == '__main__':
    import argparse
    defstr = '(default %(default)s)'
    parser = argparse.ArgumentParser()
    parser.add_argument('-wd', '--workdir', required=True)
    parser.add_argument('-r', '--run', default=None)
    args=parser.parse_args()
    workdir = args.workdir
    run = args.run

def get_pars():
    datadir = '/om/project/bild/Analysis/task_openfmri/openfmri/'
    pars = [x for x in os.listdir(datadir) if 'BILD' in x]
    for i in pars:
         if 'DTI' in i or '_2' in i:
             pars.pop(pars.index(i))
    return pars


def limits(pars):
    chunks = []
    if len(pars) > 20:
        for i in pars[:20]:
            chunks.append(i)
    else:
        chunks = pars
    return chunks


def pars_to_run(specdir):
    pars = np.genfromtxt(os.path.join('/om/project/bild/Analysis/task_openfmri/scripts/fmri',
                         'participants_to_run_firstlevel.txt'), 
                         delimiter='\n', dtype=str)
    pars = list(pars)
    wdir = '/om/project/bild/Analysis/task_openfmri/first_level'
    for i in pars:
        if os.path.isdir(os.path.join(wdir,specdir,i)):
            pars.pop(pars.index(i))
    return pars

def command(par):
    cmd_args = """#!/bin/bash
#SBATCH --mem=10G
#SBATCH -c2
#SBATCH --time=2-23:00:00\n""".format(par)
    
    command = """
    for task in 1 2; do python /om/project/bild/Analysis/task_openfmri/scripts/fmri/fmri_ants_openfmri_sparse_3mm.py \
    --sd /mindhive/xnat/surfaces/BILD/Kids \
    --target /om/project/bild/Analysis/task_openfmri/misc/OASIS-30_Atropos_template_in_MNI152_2mm.nii.gz \
    -d /om/project/bild/Analysis/task_openfmri/openfmri \
    -t {} -m 1 -x "BILD*" -s {} \
    -w /om/project/bild/Analysis/task_openfmri/first_level/working_dir_071616/{} \
    -o /om/project/bild/Analysis/task_openfmri/first_level/output_dir_071616 -p "SLURM"; done
    """.format('$task',par,par,par)
    
    cmd = cmd_args +  ' '.join(command.split())
    return cmd.split('\n')

def npwrite(text, par):
    np.savetxt('{}_firstlevel.sh'.format(par), text, delimiter='\n', fmt="%s")

os.chdir('/om/project/bild/Analysis/task_openfmri/scripts/fmri/bash_files')

if run == 'all':
    for i in get_pars():
        npwrite(command(i), i)
elif run == None:
    for i in limits(pars_to_run(workdir)):
        npwrite(command(i), i)

runs = [x for x in os.listdir('.') if '_firstlevel.sh' in x]
slurms = [x for x in os.listdir('.') if 'slurm-' in x]

for q in runs:
    subprocess.call(['sbatch',os.path.join(os.getcwd(),q)])
    shutil.move(os.path.join(os.getcwd(), q),
                "/om/project/bild/Analysis/task_openfmri/scripts/fmri/bash_files/ran")

for k in slurms:
    shutil.move(os.path.join(os.getcwd(), k),
                "/om/project/bild/Analysis/task_openfmri/scripts/fmri/bash_files/slurms_ran")
