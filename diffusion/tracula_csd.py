import os
from glob import glob

from nipype import config
config.enable_provenance()

from nipype import Node, Function, Workflow, IdentityInterface
from nipype.interfaces.io import SelectFiles, DataSink

import os
from glob import glob

brun = 'dwi_1k'

data_dir = '/om/project/voice/processedData/openfmri'
tracula_dir = '/om/project/voice/processedData/tracula/'#+brun
sids = ['GAB_ISN_012','GAB_ISN_014','GAB_ISN_017']

if not os.path.exists(tracula_dir):
    os.mkdir(tracula_dir)


def run_prep(sid, template, data_dir,brun):
    from glob import glob
    import os
    #nifti = os.path.abspath(glob(os.path.join(data_dir, '%s/diff/preproc/mri/diff_preproc.nii.gz' % sid))[0])
    #bvec = os.path.abspath(glob(os.path.join(data_dir, '%s/diff/preproc/bvecs_fsl_moco_norm.txt' % sid))[0])
    #bval = os.path.abspath(glob(os.path.join(data_dir, '%s/diff/preproc/bvals.txt' % sid))[0])
    nifti = os.path.abspath(glob(os.path.join(data_dir, '%s/dmri/preproc/%s_disco.nii.gz' % (sid,brun)))[0])
    bvec = os.path.abspath(glob(os.path.join(data_dir, '%s/dmri/%s.bvecs' % (sid,brun)))[0])
    bval = os.path.abspath(glob(os.path.join(data_dir, '%s/dmri/%s.bvals' % (sid,brun)))[0])
    from string import Template
    with open(template, 'rt') as fp:
        tpl = Template(fp.read())
    out = tpl.substitute(subjects=sid, bvec=bvec, bval=bval, niftis=nifti)
    config_file = os.path.join(os.getcwd(), 'config_%s' % sid)
    with open(config_file, 'wt') as fp:
        fp.write(out)
    from nipype import config
    config.enable_provenance()
    from nipype.interfaces.base import CommandLine
    from nipype.pipeline.engine import Node
    node = Node(CommandLine('trac-all -prep -c %s -no-isrunning -noqa' % config_file, terminal_output='allatonce'),
                name='trac-prep-%s' % sid)
    node.base_dir = os.getcwd()
    node.run()
    return sid, config_file

def run_bedpost(sid, tracula_dir):
    import os
    from nipype import config
    config.enable_provenance()
    from nipype.interfaces.base import CommandLine
    pwd = os.getcwd()
    os.chdir(os.path.join(tracula_dir, sid))
    bedpost = CommandLine('bedpostx_gpu dmri --model=2 --rician', terminal_output='allatonce')
    bedpost.run()
    os.chdir(pwd)
    return sid

def run_path(sid, config_file):
    import os
    from nipype import config
    config.enable_provenance()
    from nipype.interfaces.base import CommandLine
    from nipype.pipeline.engine import Node
    node = Node(CommandLine('trac-all -path -c %s -no-isrunning' % config_file, terminal_output='allatonce'),
                name='trac-path-%s' % sid)
    node.base_dir = os.getcwd()
    node.run()
    return sid

'''
def run_ptx2(sid, tracula_dir):
    import os
    from nipype import config
    config.enable_provenance()
    from nipype.interfaces.base import CommandLine
    import nipype.interfaces.fsl as fsl
    pwd = os.getcwd()
    os.chdir(os.path.join(tracula_dir, sid))

    # make sure have good regions:
    #

    ptx2 = pe.MapNode(fsl.ProbTrackX2,name='ptx2',iterfield=["seed"])
'''
def dmri_recon(sid, tracula_dir, brun, recon='csd', num_threads=1):
    import tempfile
    tempfile.tempdir = '/om/scratch/Fri/ksitek/'

    import os
    oldval = None
    if 'MKL_NUM_THREADS' in os.environ:
        oldval = os.environ['MKL_NUM_THREADS']
    os.environ['MKL_NUM_THREADS'] = '%d' % num_threads
    ompoldval = None
    if 'OMP_NUM_THREADS' in os.environ:
        ompoldval = os.environ['OMP_NUM_THREADS']
    os.environ['OMP_NUM_THREADS'] = '%d' % num_threads
    import nibabel as nib
    import numpy as np
    from glob import glob


    fimg = os.path.abspath(glob(os.path.join(tracula_dir, '%s/dmri/dwi.nii.gz' % sid))[0])
    fbvec = os.path.abspath(glob(os.path.join(tracula_dir, '%s/dmri/bvecs' % sid))[0])
    fbval = os.path.abspath(glob(os.path.join(tracula_dir, '%s/dmri/bvals' % sid))[0])
    img = nib.load(fimg)
    data = img.get_data()

    prefix = sid

    from dipy.io import read_bvals_bvecs
    from dipy.core.gradients import vector_norm
    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
    b0idx = []
    for idx, val in enumerate(bvals):
        if val < 1:
            pass
            #bvecs[idx] = [1, 0, 0]
        else:
            b0idx.append(idx)
    bvecs[b0idx, :] = bvecs[b0idx, :]/vector_norm(bvecs[b0idx])[:, None]

    from dipy.core.gradients import gradient_table
    gtab = gradient_table(bvals, bvecs)

    from dipy.reconst.csdeconv import auto_response
    response, ratio = auto_response(gtab, data, roi_radius=10, fa_thr=0.7)

    #from dipy.segment.mask import median_otsu
    #b0_mask, mask = median_otsu(data[:, :, :, b0idx].mean(axis=3).squeeze(), 4, 4)

    fmask1 = os.path.abspath(glob(os.path.join(tracula_dir,
                                              '%s/dlabel/diff/aparc+aseg_mask.bbr.nii.gz' % sid))[0])
    fmask2 = os.path.abspath(glob(os.path.join(tracula_dir,
                                              '%s/dlabel/diff/notventricles.bbr.nii.gz' % sid))[0])
    mask = (nib.load(fmask1).get_data() > 0.5) * nib.load(fmask2).get_data()

    useFA = True
    if recon == 'csd':
        from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
        model = ConstrainedSphericalDeconvModel(gtab, response)
        useFA = True
    elif recon == 'csa':
        from dipy.reconst.shm import CsaOdfModel, normalize_data
        model = CsaOdfModel(gtab, 4)
        useFA = False
    else:
        raise ValueError('only csd, csa supported currently')
        from dipy.reconst.dsi import (DiffusionSpectrumDeconvModel,
                                      DiffusionSpectrumModel)
        model = DiffusionSpectrumDeconvModel(gtab)
    #fit = model.fit(data)

    from dipy.data import get_sphere
    sphere = get_sphere('symmetric724')
    #odfs = fit.odf(sphere)

    from dipy.reconst.peaks import peaks_from_model
    peaks = peaks_from_model(model=model,
                             data=data,
                             sphere=sphere,
                             mask=mask,
                             return_sh=True,
                             return_odf=False,
                             normalize_peaks=True,
                             npeaks=5,
                             relative_peak_threshold=.5,
                             min_separation_angle=25,
                             parallel=num_threads > 1,
                             nbr_processes=num_threads)

    from dipy.reconst.dti import TensorModel
    tenmodel = TensorModel(gtab)
    tenfit = tenmodel.fit(data, mask)

    from dipy.reconst.dti import fractional_anisotropy
    FA = fractional_anisotropy(tenfit.evals)
    FA[np.isnan(FA)] = 0
    fa_img = nib.Nifti1Image(FA, img.get_affine())
    tensor_fa_file = os.path.abspath('%s_%s_tensor_fa.nii.gz' % (prefix, brun))
    nib.save(fa_img, tensor_fa_file)

    evecs = tenfit.evecs
    evec_img = nib.Nifti1Image(evecs, img.get_affine())
    tensor_evec_file = os.path.abspath('%s_%s_tensor_evec.nii.gz' % (prefix, brun))
    nib.save(evec_img, tensor_evec_file)

    #from dipy.reconst.dti import quantize_evecs
    #peak_indices = quantize_evecs(tenfit.evecs, sphere.vertices)
    #eu = EuDX(FA, peak_indices, odf_vertices = sphere.vertices, a_low=0.2, seeds=10**6, ang_thr=35)

    fa_img = nib.Nifti1Image(peaks.gfa, img.get_affine())
    model_gfa_file = os.path.abspath('%s_%s_%s_gfa.nii.gz' % (prefix, brun, recon))
    nib.save(fa_img, model_gfa_file)

    from dipy.tracking.eudx import EuDX
    if useFA:
        eu = EuDX(FA, peaks.peak_indices[..., 0], odf_vertices = sphere.vertices,
                  a_low=0.1, seeds=10**6, ang_thr=35)
    else:
        eu = EuDX(peaks.gfa, peaks.peak_indices[..., 0], odf_vertices = sphere.vertices,
                  a_low=0.1, seeds=10**6, ang_thr=35)

    #import dipy.tracking.metrics as dmetrics
    streamlines = ((sl, None, None) for sl in eu) # if dmetrics.length(sl) > 15)

    hdr = nib.trackvis.empty_header()
    hdr['voxel_size'] = fa_img.get_header().get_zooms()[:3]
    hdr['voxel_order'] = 'LAS'
    hdr['dim'] = FA.shape[:3]

    sl_fname = os.path.abspath('%s_%s_%s_streamline.trk' % (prefix, brun, recon))

    nib.trackvis.write(sl_fname, streamlines, hdr, points_space='voxel')
    if oldval:
        os.environ['MKL_NUM_THREADS'] = oldval
    else:
        del os.environ['MKL_NUM_THREADS']
    if ompoldval:
        os.environ['OMP_NUM_THREADS'] = ompoldval
    else:
        del os.environ['OMP_NUM_THREADS']
    return tensor_fa_file, tensor_evec_file, model_gfa_file, sl_fname

infosource = Node(IdentityInterface(fields=['subject_id','brun']), name='infosource')
infosource.iterables = ('subject_id', sids)
infosource.inputs.brun = brun

'''
# freesurfer
from nipype.interfaces.freesurfer.preprocess import ReconAll
subjects_dir = os.path.abspath('/om/user/ksitek/HCP/fsdata/')
recon_all = pe.MapNode(interface=ReconAll(), name='recon_all',
                       iterfield=['subject_id', 'T1_files'])
recon_all.inputs.subject_id = subject_list
if not os.path.exists(subjects_dir):
    os.mkdir(subjects_dir)
recon_all.inputs.subjects_dir = subjects_dir

wf.connect(datasource, 'struct', recon_all, 'T1_files')
'''

node1 = Node(Function(input_names=['sid', 'template', 'data_dir','brun'],
                     output_names=['sid', 'config_file'], function=run_prep),
            name='trac-prep')
node1.inputs.template = os.path.abspath('tracula_config_%s'%brun)
node1.inputs.data_dir = data_dir

node2 = Node(Function(input_names=['sid', 'tracula_dir'],
                     output_names=['sid'], function=run_bedpost),
            name='trac-bedp')
node2.inputs.tracula_dir = tracula_dir
node2.plugin_args = {'sbatch_args': '--gres=gpu:1',
                      'overwrite': True}

node3 = Node(Function(input_names=['sid', 'config_file'],
                     output_names=['sid'], function=run_path),
            name='trac-path')
'''
probtx = Node(Function(input_names=['sid','tracula_dir'],
                        output_names=['sid'],function=run_ptx2),
                        name='probtx')
probtx.inputs.tracula_dir = tracula_dir
probtx.inputs.opd=True
probtx.inputs.os2t=True
probtx.plugin_args = {'sbatch_args': '-p om_all_nodes --mem=16384 -N1 -c1','max_jobs':300,
                      'overwrite': True}
'''
tracker = Node(Function(input_names=['sid', 'tracula_dir', 'brun','recon', 'num_threads'],
                        output_names=['tensor_fa_file', 'tensor_evec_file', 'model_gfa_file',
                                      'model_track_file'],
                        function=dmri_recon), name='tracker')
tracker.inputs.recon = 'csd'
tracker.inputs.tracula_dir = tracula_dir
num_threads = 20
tracker.inputs.num_threads = num_threads
tracker.plugin_args = {'sbatch_args': '-p om_all_nodes --mem=%dG -N 1 -c %d' % (10 * num_threads, #3 * num_th
                                                                                    num_threads),
                       'overwrite': True}

ds = Node(DataSink(parameterization=False), name='sinker')
ds.inputs.base_directory = tracula_dir
ds.plugin_args = {'overwrite': True}

wf = Workflow(name='isn-tracula_%s'%brun)

wf.connect(infosource, 'subject_id', node1, 'sid')
wf.connect(infosource, 'brun', node1, 'brun')

#wf.connect(infosource, 'subject_id', tracker, 'sid')
#wf.connect(infosource, 'subject_id', ds, 'container')

wf.connect(node1, 'sid', node2, 'sid')
#wf.connect(infosource, 'subject_id', node2, 'sid')
wf.connect(node1, 'sid', tracker, 'sid')
#wf.connect(node2, 'sid', tracker, 'sid')

wf.connect(infosource, 'brun', tracker, 'brun')
wf.connect(node2, 'sid', node3, 'sid')

wf.connect(node1, 'config_file', node3, 'config_file')
wf.connect(node1, 'sid', ds, 'container')

wf.connect(tracker, 'tensor_fa_file', ds, 'recon.@fa')
wf.connect(tracker, 'tensor_evec_file', ds, 'recon.@evec')
wf.connect(tracker, 'model_gfa_file', ds, 'recon.@gfa')
wf.connect(tracker, 'model_track_file', ds, 'recon.@track')

wf.base_dir = '/om/scratch/Fri/ksitek/isn-%s'%brun

wf.run(plugin='SLURM', plugin_args={'sbatch_args': '--mem=5G -N1 -c2'})
