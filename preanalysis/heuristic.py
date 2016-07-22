import os

def create_key(template, outtype=('nii.gz',), annotation_classes=None):
    if template is None or not template:
        raise ValueError('Template must be a valid format string')
    return (template, outtype, annotation_classes)

def infotodict(seqinfo):
    """Heuristic evaluator for determining which runs belong where
    
    allowed template fields - follow python string module: 
    
    item: index within category 
    subject: participant id 
    seqitem: run number during scanning
    subindex: sub index within group
    """
    
    rs = create_key('resting/rest/bold', outtype=('dicom', 'nii.gz'))
    dwi = create_key('dmri/dwi_{item:03d}', outtype=('dicom','nii.gz'))
    t1 = create_key('anatomy/T1_{item:03d}')   
    t2 = create_key('anatomy/T2_{item:03d}')
    nonwordrep = create_key('BOLD/task001_run{item:03d}/bold')
    syntax=create_key('BOLD/task002_run{item:03d}/bold')
    fm = create_key('field_map/fm_{item:03d}')
    info = {rs: [], dwi:[], t1:[], t2:[], nonwordrep:[], syntax:[], fm:[]}
    last_run = len(seqinfo)
    for s in seqinfo:
        x,y,sl,nt = (s[6], s[7], s[8], s[9])
        if ('Rest' in s[12]):
            info[rs] = [s[2]]
        elif (sl == 74) and (nt == 40) and ('DIFFUSION' in s[12]):
            info[dwi].append(s[2])
        elif (sl == 176) and (nt ==1) and ('T1_MPRAGE' in s[12]):
            info[t1].append(s[2])
        elif (nt == 42) and ('Nonword' in s[12]):
            if not s[13]:
	        info[nonwordrep].append(s[2])
        elif (nt == 50) and ('Syntax' in s[12]):
            if not s[13]:
                info[syntax].append(s[2])
        elif (nt == 1) and ('field_mapping' in s[12]):
            info[fm].append(s[2])
        elif (sl == 176) and (nt==1) and ('T2_SPACE' in s[12]):
            info[t2] = [s[2]]  
	else:
            pass
    return info
