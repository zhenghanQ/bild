# export QT_API=pyqt NO LONGER NEED THIS
# make sure you've done: pip install https://github.com/satra/nibabel/archive/enh/cifti2.zip
# xvfb-run --server-args="-screen 0 1024x768x24" python py_plots_surfs.py  THIS NO LONGER WORKS ON OPENMIND7


######################################################
#   KEEP THIS FIRST TO AVOID QT_API SET TO 1 ERROR   #
######################################################
import sip
sip.setapi('QDate', 2)
sip.setapi('QString', 2)
sip.setapi('QTextStream', 2)
sip.setapi('QTime', 2)
sip.setapi('QUrl', 2)
sip.setapi('QVariant', 2)
sip.setapi('QDateTime', 2)
######################################################
######################################################

import matplotlib.pyplot as plt
import os
from glob import glob
import numpy as np
import nibabel as nb
import nibabel.gifti as gifti
from mayavi import mlab
from tvtk.api import tvtk
import math


img = nb.load('/om/user/ysa/rfMRI_REST1_LR_Atlas.dtseries.nii') 
mim = img.header.matrix.mims[1]
bm1 = mim.brain_models[0]
lidx = bm1.vertex_indices.indices
bm2 = mim.brain_models[1]
ridx = bm1.surface_number_of_vertices + bm2.vertex_indices.indices
bidx = np.concatenate((lidx, ridx))


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    theta = np.asarray(theta)
    axis = axis/math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2)
    b, c, d = -axis*math.sin(theta/2)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

axis = [0, 0, 1]
theta = np.pi

inflated = True
split_brain = True

surf = gifti.read('/om/user/ysa/brain_plots/32k_ConteAtlas_v2/Conte69.L.midthickness.32k_fs_LR.surf.gii') #inflated.32k_fs_LR.surf.gii')

verts_L_data = surf.darrays[0].data
faces_L_data = surf.darrays[1].data

surf = gifti.read('/om/user/ysa/brain_plots/32k_ConteAtlas_v2/Conte69.R.midthickness.32k_fs_LR.surf.gii') #inflated.32k_fs_LR.surf.gii')
verts_R_data = surf.darrays[0].data
faces_R_data = surf.darrays[1].data

if inflated:
    surf = gifti.read('/om/user/ysa/brain_plots/32k_ConteAtlas_v2/Conte69.L.inflated.32k_fs_LR.surf.gii')
    # Conte69.L.midthickness.32k_fs_LR.surf.gii for less inflated, left side
    verts_L_display = surf.darrays[0].data
    faces_L_display = surf.darrays[1].data
    surf = gifti.read('/om/user/ysa/brain_plots/32k_ConteAtlas_v2/Conte69.R.inflated.32k_fs_LR.surf.gii')
    # Conte69.R.midthickness.32k_fs_LR.surf.gii for less inflated, right side
    verts_R_display = surf.darrays[0].data
    faces_R_display = surf.darrays[1].data
else:
    verts_L_display = verts_L_data.copy()
    verts_R_display = verts_R_data.copy()
    faces_L_display = faces_L_data.copy()
    faces_R_display = faces_R_data.copy()

verts_L_display[:, 0] -= max(verts_L_display[:, 0])
verts_R_display[:, 0] -= min(verts_R_display[:, 0])
verts_L_display[:, 1] -= (max(verts_L_display[:, 1]) + 1)
verts_R_display[:, 1] -= (max(verts_R_display[:, 1]) + 1)

faces = np.vstack((faces_L_display, verts_L_display.shape[0] + faces_R_display))

if split_brain:
    verts2 = rotation_matrix(axis, theta).dot(verts_R_display.T).T
else:
    verts_L_display[:, 1] -= np.mean(verts_L_display[:, 1])
    verts_R_display[:, 1] -= np.mean(verts_R_display[:, 1])
    verts2 = verts_R_display
    

verts_rot = np.vstack((verts_L_display, verts2))
verts = np.vstack((verts_L_data, verts_R_data))

def useZstat(zstat,neg_zstat):  
    img = nb.load(zstat)
    img_neg = nb.load(neg_zstat)
    # this can change your already thresholded activations patterns if set higher than 
    # what they were computed at, if set lower than the scale will change, but 
    # the patterns will remain the same
    threshold = 1.97 # 1000
    display_threshold = 6 #8000

    data = img.get_data()
    data_neg = img_neg.get_data()
    aff = img.affine
    aff_neg = img_neg.affine
    indices = np.round((np.linalg.pinv(aff).dot(np.hstack((verts, 
                                              np.ones((verts.shape[0], 1)))).T))[:3, :].T).astype(int)
    indices_neg = np.round((np.linalg.pinv(aff_neg).dot(np.hstack((verts, 
                                              np.ones((verts.shape[0], 1)))).T))[:3, :].T).astype(int)
    scalars2 = data[indices[:, 0], indices[:, 1], indices[:, 2]]
    scalars2_neg = data_neg[indices_neg[:, 0], indices_neg[:, 1], indices_neg[:, 2]]
    scalars2[np.abs(scalars2) < threshold] = 0.
    scalars2_neg[np.abs(scalars2_neg) < threshold] = 0.
    scalars = np.zeros(verts.shape[0])
    scalars_neg = np.zeros(verts.shape[0])
    scalars[bidx] = scalars2[bidx]
    scalars_neg[bidx] = scalars2_neg[bidx]
    scalars = scalars + (-scalars_neg)
    negative = positive = False
    if np.any(scalars < 0):
        negative = True
    if np.any(scalars > 0):
        positive = True
    nlabels = 2
    vmin = 0
    vmax = 0
    if negative and positive:
        maxval = max(-scalars.min(), scalars.max())
        if maxval > display_threshold:
            maxval = display_threshold

        vmin = -maxval
        vmax = maxval
        nlabels = 3
        vmin = -display_threshold ######
        vmax = display_threshold ######
    elif negative:
        vmin = scalars.min()
        if vmin < -display_threshold:
            vmin = -display_threshold
        vmax = 0
        vmin = -display_threshold ######
    elif positive:
        vmax = scalars.max()
        if vmax > display_threshold:
            vmax = display_threshold
        vmin = 0
        vmax = display_threshold #######
    print zstat
    
    # 12/8 edit
    
    dual_split = True ######
    fig1 = mlab.figure(1, bgcolor=(0, 0, 0))
    mlab.clf()
    # adds the bottom image
    mesh = tvtk.PolyData(points=verts_rot, polys=faces)
    mesh.point_data.scalars = scalars  
    mesh.point_data.scalars.name = 'scalars'
    surf = mlab.pipeline.surface(mesh, colormap='autumn', vmin=vmin, vmax=vmax)
    # adds the top image and rotates it
    if dual_split:
        verts_rot_shifted = verts_rot.copy()
        verts_rot_shifted = rotation_matrix(axis, theta).dot(verts_rot_shifted.T).T
        verts_rot_shifted[:, 2] -= (np.max(verts_rot_shifted[:, 2]) - np.min(verts_rot_shifted[:, 2]))
        verts_rot_shifted[:, 0] -= np.max(verts_rot_shifted[:, 0])
        mesh2 = tvtk.PolyData(points=verts_rot_shifted, polys=faces)
        mesh2.point_data.scalars = scalars 
        mesh2.point_data.scalars.name = 'scalars'
        surf2 = mlab.pipeline.surface(mesh2, colormap='autumn', vmin=vmin, vmax=vmax)
    colorbar = mlab.colorbar(surf, nb_labels=nlabels) #, orientation='vertical')
    lut = surf.module_manager.scalar_lut_manager.lut.table.to_array()

    if negative and positive:
        half_index = lut.shape[0] / 2
        index = int(half_index * threshold / vmax)
        lut[(half_index - index + 1):(half_index + index), :] = 192
        lut[(half_index + index):, :] = 255 * plt.cm.autumn(np.linspace(0, 255, half_index - index).astype(int))
        lut[:(half_index - index), :] = 255 * plt.cm.Blues(np.linspace(0, 255, half_index - index).astype(int))
    elif negative:
        index =  int(lut.shape[0] * threshold / abs(vmin))
        lut[(lut.shape[0] - index):, :] = 192
        lut[:(lut.shape[0] - index), :] = 255 * plt.cm.Blues(np.linspace(0, 255, lut.shape[0] - index).astype(int))
    elif positive:
        index = int(lut.shape[0] * threshold / vmax)
        lut[:index, :] = 192
        lut[index:, :] = 255 * plt.cm.autumn(np.linspace(0, 255, lut.shape[0] - index).astype(int))
    lut[:, -1] = 255
    surf.module_manager.scalar_lut_manager.lut.table = lut
    if dual_split:
        surf2.module_manager.scalar_lut_manager.lut.table = lut
    surf.module_manager.scalar_lut_manager.show_scalar_bar = False
    surf.module_manager.scalar_lut_manager.show_legend = False
    surf.module_manager.scalar_lut_manager.label_text_property.font_size = 9
    surf.module_manager.scalar_lut_manager.show_scalar_bar = True # show bar
    surf.module_manager.scalar_lut_manager.show_legend = True # show bar
    #mlab.draw()
    mlab.show()
    translate = [0, 0, 0]
    if inflated:
        zoom = -700
    else:
        zoom = -600
    if dual_split:
        if inflated:
            translate = [0,   0, -104.01467148]
        else:
            translate = [0,  0, -54.76305802]        
        if inflated:
            zoom = -750
        else:
            zoom = -570
    #mlab.view(0, 90.0, zoom, translate)
    
if __name__ == '__main__':
    import argparse
    defstr = '(default %(default)s)'
    parser = argparse.ArgumentParser()
    parser.add_argument('-posz', '--positivez', type=str, help="positive zstat")
    parser.add_argument('-negz', '--negativez', type=str, help="the reverse of the positive zstat")
    args = parser.parse_args()
    positivez = args.positivez
    negativez = args.negativez

useZstat(positivez, negativez)
