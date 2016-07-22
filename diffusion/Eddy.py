
# coding: utf-8

# In[1]:

from nipype import config
config.enable_provenance()


# In[2]:

from nipype.pipeline.engine import Node, Workflow, MapNode
from nipype.interfaces.fsl import Eddy, TOPUP, BET
from nipype.interfaces.utility import Function,IdentityInterface
from nipype.interfaces.io import SelectFiles, DataSink


# In[3]:

0.3449941351 * 92


# In[4]:

import os
from glob import glob


# In[5]:

subject_list = sorted([os.path.basename(x)[:4] for x in glob('/gablab/p/mtbi/data/S032*')])
infosource = Node(interface=IdentityInterface(fields=['subject_id']),
                     name="infosource")
infosource.iterables = ('subject_id',subject_list)

def make_PA_files(subject_id):
    import os
    import numpy as np
    from shutil import copy2
    if os.path.exists('/gablab/p/mtbi/data/%s_SESSION_1/dmri/dwi_AP001.bvals'%subject_id):

	os.makedirs('/om/user/ksitek/mTBI/bvalsbvecs/%s_SESSION_1/'%subject_id)
        copy2('/gablab/p/mtbi/data/%s_SESSION_1/dmri/dwi_AP001.bvals'%subject_id,
            '/om/user/ksitek/mTBI/bvalsbvecs/%s_SESSION_1/dwi_AP001.bvals'%subject_id)
        copy2('/gablab/p/mtbi/data/%s_SESSION_1/dmri/dwi_AP001.bvecs'%subject_id,
            '/om/user/ksitek/mTBI/bvalsbvecs/%s_SESSION_1/dwi_AP001.bvecs'%subject_id)

        bvals = np.zeros((1,10))
        bvecs = np.zeros((3,10))

        file_bvals = os.path.abspath('/om/user/ksitek/mTBI/bvalsbvecs/%s_SESSION_1/dwi_PA001.bvals'%subject_id)
        np.savetxt(file_bvals, bvals, fmt='%d')
        file_bvecs = os.path.abspath('/om/user/ksitek/mTBI/bvalsbvecs/%s_SESSION_1/dwi_PA001.bvecs'%subject_id)
        np.savetxt(file_bvecs, bvecs, fmt='%d')
    return subject_id

PA_files = Node(Function(input_names=['subject_id'],output_names=['subject_id'],
                        function=make_PA_files), name='PA_files')

templates = {'dwi':   '/gablab/p/mtbi/data/{subject_id}_SESSION_1/dmri/dwi_{PE}001.nii.gz',
             #'bvals': 'data/{subject_id}/dmri/dwi_{PE}001.bvals',
             #'bvecs': 'data/{subject_id}/dmri/dwi_{PE}001.bvecs'}
             'bvals': '/om/user/ksitek/mTBI/bvalsbvecs/{subject_id}_SESSION_1/dwi_{PE}001.bvals',
             'bvecs': '/om/user/ksitek/mTBI/bvalsbvecs/{subject_id}_SESSION_1/dwi_{PE}001.bvecs'}
PE_order = ['AP', 'PA']
select = MapNode(SelectFiles(templates), iterfield=['PE'], name='select')
select.inputs.base_directory = os.path.abspath('/om/user/ksitek/mTBI/')
#select.inputs.bval = '2k'
select.inputs.PE = PE_order
#res = select.run()
#print res.outputs


# In[6]:

def create_files(in_files, bval_files, bvec_files, order):
    import numpy as np
    import os
    from nipype.interfaces.fsl import Merge, Split
    bvecs = []
    bvals = []
    indices = []
    acqparams = []
    b0indices = []
    for idx, fname in enumerate(bval_files):
        vals = np.genfromtxt(fname).flatten()
        bvals.extend(vals.tolist())
        b0idx = np.nonzero(vals==0)[0]
        b0indices.extend(len(indices) + b0idx)
        index = np.zeros(vals.shape)
        index[b0idx] = 1
        index = np.cumsum(index)
        indices.extend(len(acqparams) + index)
        acqp = dict(AP=[0, -1, 0, 0.032], PA=[0, 1, 0, 0.032])[order[idx]]
        for _ in range(len(b0idx)):
            acqparams.append(acqp)
        vals = np.genfromtxt(bvec_files[idx])
        if vals.shape[0] == 3:
            vals = vals.T
        bvecs.extend(vals.tolist())
    merged_file = os.path.join(os.getcwd(), 'merged.nii.gz')
    Merge(in_files=in_files, dimension='t', output_type='NIFTI_GZ', merged_file=merged_file).run()
    merged_bvals = os.path.join(os.getcwd(), 'merged.bvals')
    np.savetxt(merged_bvals, bvals, '%.1f')
    merged_bvecs = os.path.join(os.getcwd(), 'merged.bvecs')
    np.savetxt(merged_bvecs, bvecs, '%.10f %.10f %.10f')
    merged_index = os.path.join(os.getcwd(), 'merged.index')
    np.savetxt(merged_index, indices, '%d')
    acq_file = os.path.join(os.getcwd(), 'b0_acq.txt')
    np.savetxt(acq_file, acqparams, '%d %d %d %f')
    b0file = os.path.join(os.getcwd(), 'b0_merged.nii.gz')
    res = Split(in_file=merged_file, dimension='t').run()
    Merge(in_files=np.array(res.outputs.out_files)[b0indices].tolist(), dimension='t',
          output_type='NIFTI_GZ', merged_file=b0file).run()
    return merged_file, merged_bvals, merged_bvecs, merged_index, acq_file, b0file

#preproc.inputs.in_files = res.outputs.dwi
#preproc.inputs.bval_files = res.outputs.bvals
#preproc.inputs.bvec_files = res.outputs.bvecs
#preproc.run()


# In[7]:

def rotate_bvecs(bvec_file, par_file):
    import os
    import numpy as np
    pars = np.genfromtxt(par_file)
    bvecs = np.genfromtxt(bvec_file)
    new_bvecs = []
    rotfunc = lambda x: np.array([[np.cos(x), np.sin(x)],
                                  [-np.sin(x), np.cos(x)]])
    for idx, vector in enumerate(bvecs):
        par = pars[idx]
        Rx = np.eye(3)
        Rx[1:3, 1:3] = rotfunc(par[3])
        Ry = np.eye(3)
        Ry[(0, 0, 2, 2), (0, 2, 0, 2)] = rotfunc(par[4]).ravel()
        Rz = np.eye(3)
        Rz[0:2, 0:2] = rotfunc(par[5])
        R = np.linalg.inv(Rx.dot(Ry.dot(Rz)))
        new_bvecs.append(R.dot(vector.T).tolist())
    new_bvec_file = os.path.join(os.getcwd(), 'rotated.bvecs')
    np.savetxt(new_bvec_file, new_bvecs, '%.10f %.10f %.10f')
    return new_bvec_file


# In[8]:

def dmri_recon(fimg, fbval, fbvec, recon='csd', prefix='recon'):
    import os
    import nibabel as nib
    import numpy as np
    img = nib.load(fimg)
    data = img.get_data()

    num_threads = 4

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

    from dipy.segment.mask import median_otsu
    b0_mask, mask = median_otsu(data[:, :, :, b0idx].mean(axis=3).squeeze(), 4, 4)

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
                             parallel=True)

    from dipy.reconst.dti import TensorModel
    tenmodel = TensorModel(gtab)
    tenfit = tenmodel.fit(data, mask)

    from dipy.reconst.dti import fractional_anisotropy
    FA = fractional_anisotropy(tenfit.evals)
    FA[np.isnan(FA)] = 0
    fa_img = nib.Nifti1Image(FA, img.get_affine())
    tensor_fa_file = os.path.abspath('%s_tensor_fa.nii.gz' % prefix)
    nib.save(fa_img, tensor_fa_file)

    evecs = tenfit.evecs
    evec_img = nib.Nifti1Image(evecs, img.get_affine())
    tensor_evec_file = os.path.abspath('%s_tensor_evec.nii.gz' % prefix)
    nib.save(evec_img, tensor_evec_file)

    #from dipy.reconst.dti import quantize_evecs
    #peak_indices = quantize_evecs(tenfit.evecs, sphere.vertices)
    #eu = EuDX(FA, peak_indices, odf_vertices = sphere.vertices, a_low=0.2, seeds=10**6, ang_thr=35)

    fa_img = nib.Nifti1Image(FA, img.get_affine())
    model_gfa_file = os.path.abspath('%s_%s_gfa.nii.gz' % (prefix, recon))
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

    sl_fname = os.path.abspath('%s_%s_streamline.trk' % (prefix, recon))

    nib.trackvis.write(sl_fname, streamlines, hdr, points_space='voxel')
    return tensor_fa_file, tensor_evec_file, model_gfa_file, sl_fname


# In[9]:

wf = Workflow(name='eddy')

preproc = Node(Function(input_names=['in_files', 'bval_files', 'bvec_files', 'order'],
                        output_names=['merged_file', 'merged_bvals', 'merged_bvecs', 'merged_index',
                                      'acq_file', 'b0file'],
                        function=create_files), name='preproc')
preproc.inputs.order = PE_order

wf.connect(infosource, 'subject_id', PA_files, 'subject_id')

wf.connect(PA_files, 'subject_id', select, 'subject_id')

wf.connect([(select, preproc, [('dwi', 'in_files'),
                               ('bvals', 'bval_files'),
                               ('bvecs', 'bvec_files')])])

topup = Node(TOPUP(out_corrected='b0correct.nii.gz', numprec='float', output_type='NIFTI_GZ'), name='topup')
wf.connect(preproc, 'acq_file', topup, 'encoding_file')
wf.connect(preproc, 'b0file', topup, 'in_file')
masker = Node(BET(mask=True), name='mask')
wf.connect(topup, 'out_corrected', masker, 'in_file')

eddy = Node(Eddy(), name='eddy')
eddy.inputs.num_threads = 4
eddy.plugin_args = {'sbatch_args': '--mem=10G -c 4'}

wf.connect(masker, 'mask_file', eddy, 'in_mask')
wf.connect(preproc, 'merged_file', eddy, 'in_file')
wf.connect(preproc, 'merged_bvals', eddy, 'in_bval')
wf.connect(preproc, 'merged_bvecs', eddy, 'in_bvec')
wf.connect(preproc, 'merged_index', eddy, 'in_index')
wf.connect(preproc, 'acq_file', eddy, 'in_acqp')
wf.connect(topup, 'out_fieldcoef', eddy, 'in_topup_fieldcoef')
wf.connect(topup, 'out_movpar', eddy, 'in_topup_movpar')

rotate = Node(Function(input_names=['bvec_file', 'par_file'],
                        output_names=['bvec_file'],
                        function=rotate_bvecs), name='rotate')

wf.connect(preproc, 'merged_bvecs', rotate, 'bvec_file')
wf.connect(eddy, 'out_parameter', rotate, 'par_file')

tracker = Node(Function(input_names=['fimg', 'fbval', 'fbvec', 'recon', 'prefix'],
                        output_names=['tensor_fa_file', 'tensor_evec_file', 'model_gfa_file',
                                      'model_track_file'],
                        function=dmri_recon), name='tracker')
#tracker.inputs.recon = 'csa'
#tracker.inputs.prefix = 'pilot009'
tracker.plugin_args = {'sbatch_args': '--mem=50G -c 4'}

wf.connect(eddy, 'out_corrected', tracker, 'fimg')
wf.connect(preproc, 'merged_bvals', tracker, 'fbval')
wf.connect(rotate, 'bvec_file', tracker, 'fbvec')
wf.connect(infosource,'subject_id',tracker,'prefix')

ds = Node(DataSink(), name='sinker')
ds.inputs.base_directory = os.path.join(os.getcwd(), 'outputs')
wf.connect(PA_files,'subject_id',ds,'container')
wf.connect(preproc, 'merged_bvecs', ds, 'pre.@bvec_file')
wf.connect(preproc, 'merged_bvals', ds, '@bval_file')
wf.connect(rotate, 'bvec_file', ds, '@rot_bvec_file')
wf.connect(eddy, 'out_corrected', ds, '@out_file')
wf.connect(tracker, 'tensor_fa_file', ds, 'recon.@fa')
wf.connect(tracker, 'tensor_evec_file', ds, 'recon.@evec')
wf.connect(tracker, 'model_gfa_file', ds, 'recon.@gfa')
wf.connect(tracker, 'model_track_file', ds, 'recon.@track')


# In[ ]:

## RUN
wf.base_dir = os.path.abspath('/om/user/ksitek/mTBI')
wf.run('SLURM', plugin_args={'sbatch_args': '--qos=gablab -N1 -c2 --mem=10G'})
