
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import os
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pylab
from nilearn.plotting import plot_stat_map
import nipype.interfaces.utility as util
import nipype.pipeline.engine as pe
import nipype.interfaces.io as nio
import numpy as np


# In[3]:


#Project Vars:
#data_dir=os.path.abspath('/om/user/cdla/projects/bild/openfmri/')
data_dir=os.path.abspath('/Users/cdla/Desktop/bild/')
l1_dir=os.path.join(data_dir, 'l1output')
subj_prefix= 'BILD*'
fs_dir=os.path.join(data_dir,'surfaces')
template=os.path.join(data_dir,'scripts')
data_dir=os.path.join(data_dir,'openfmri')
model=1
task=1
runs=[0,1,2]


# In[4]:


task_key = os.path.join(data_dir, 'task_key.txt')
task_contrasts = os.path.join(data_dir, 'models', 'model%03d'%model, 'task_contrasts.txt')
condition_key = os.path.join(data_dir, 'models', 'model%03d'%model, 'condition_key.txt')


# In[5]:


def get_tasks(l1_dir,task_key):
    with open(task_key, 'r') as f:
        tasks = [line.split()[0] for line in f]
    for task in tasks:
        if os.path.exists(os.path.join(l1_dir,'model%03d'%model,'task%03d'%task)) != True:
            tasks = sorted(os.listdir(os.path.join(l1_dir,'model%03d'%model)))
    return tasks

def get_subjlist(l1_dir,subj_prefix,model, task):
    subj_list= [os.path.basename(x) for x in sorted(glob(os.path.join(l1_dir,'model%03d'%model,'task%03d'%task,subj_prefix)))]
    return subj_list

def get_datalist(l1_dir,subj_prefix):
    subj_list = [os.path.basename(x) for x in sorted(glob(os.path.join(data_dir,subj_prefix)))]
    return subj_list

def get_runs(task, subj_list):
    runs_list = []
    for subj in subjlist:
        task_runs = sorted(os.listdir(os.path.join(openfmri_dir,subj,'BOLD')))
        num_runs = 0
        for run in task_runs:
            if run.split('_')[0] == task:
                num_runs += 1
        runs_list.append(num_runs)
    return runs_list

def get_num_vols(bold_file):
    bold_img = nb.load(bold_file)
    num_vols = bold_img.shape[3]
    return num_vols

def plot_contrasts(data_dir,l1_dir,model,task,subj):
    anat=os.path.join(data_dir,subj,'anatomy','T1_001.nii.gz')
    contrasts=sorted(glob(os.path.join(l1_dir,'model%03d'%model,'task%03d'%task,subj,'zstat','zstat*')))
    for contrast in contrasts:
        plot_stat_map(contrast, anat, threshold=1.6, title=contrast)
        print 'test'
        plt.show()


# In[6]:


#print finished subjs
#print unfinished subjs
finished_subj_list=get_subjlist(l1_dir,subj_prefix,model,task)
total_subjs=get_datalist(l1_dir,subj_prefix)
unfinished_subjs=[x for x in total_subjs if x not in finished_subj_list]
print 'there are %d subjects in the data_dir'%len(total_subjs)
print 'there are %d finished subjects.'%len(finished_subj_list)
print 'there are %d unfinished subjects.'%len(unfinished_subjs)
print 'Unfinished subjs are:'
print unfinished_subjs
subj_list=finished_subj_list


# In[7]:


#visualize model
for subj in [subj_list[0]]:
    for run in runs:
        model_files=sorted(glob(os.path.join(l1_dir,'model%03d'%model,'task%03d'%task,subj,'qa',
                                        'model','model%03d'%model,'task%03d'%task,'run%02d_run%01d.png'%(run+1,run))))
        models=pylab.figure()
        for idx,model_file in enumerate(model_files):
            img=mpimg.imread(model_file)
            plt.imshow(img)
            plt.axis('off')
            plt.title('model for run %s for subj %s'%(idx,subj))
            idx=idx+1


# In[12]:


#visualize contrasts:
def plot_contrasts(data_dir,l1_dir,model,task,subj):
    anat=os.path.join(data_dir,subj,'anatomy','T1_001.nii.gz')
    contrasts=sorted(glob(os.path.join(l1_dir,'model%03d'%model,'task%03d'%task,subj,'zstats','zstat*')))
    for contrast in contrasts:
        plot_stat_map(contrast, anat, threshold=2.3, title=contrast, display_mode='z', cut_coords=7)
        plot_stat_map(contrast, anat, threshold=2.3, title=contrast, display_mode='x', cut_coords=7)
        plot_stat_map(contrast, anat, threshold=2.3, title=contrast, display_mode='y', cut_coords=7)
        plt.show()
for subj in [subj_list[0]]:    
    plot_contrasts(data_dir,l1_dir,model,task,subj)


# In[9]:


#stim_corr
def stim_corr(subinfo, inpath, sparse, subject_id):
    import scipy as scipy
    import scipy.io as sio
    from scipy.stats.stats import pearsonr
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import glob

    alls = []
    out_images1 = []
    out_images2 = []
    output_info = []
    
    if not sparse:
        for j, i in enumerate(subinfo):
            c_label = i.conditions
            cond = len(i.conditions)
            # reg = len(i.regressor_names)
        output_path = os.path.abspath("Outlier.csv")
        ofile = open(output_path, 'w')
        ofile.write(', '.join(["Subject ID"]+["Run"]+["Outlier All"]+["Outlier in %s" %c_label[d] for d in range(cond)]))
        ofile.write('\n')
        for r in range(len(subinfo)): 
            run = 'run%s' %(r)
            # path = os.path.join(inpath, '_generate_model%d/run%d.mat' %(r,r)) #
            if len(subinfo) > 1:
                param = np.genfromtxt(inpath[r], skip_header=5)
            else:
                param = np.genfromtxt(inpath, skip_header=5)
            mat = param.shape
            columns = param.shape[1]
            count = cond+6
            outlier = columns  count
            out = 'Outlier = %d' %(outlier)
            con = 'Conditions = %d' %(cond)
            matr = 'Design Matrix Shape = [%d rows, %d columns]' %(mat)
            output_info.append([[run, out, con, matr]])
            ofile.write(', '.join([str(subject_id)]+[str(r)]+[str(outlier)]))
            if outlier > 0:
                o = param[:, count:columns]
                o_sums = o.sum(axis=1)
                param_o = np.column_stack((param, o_sums))
                # param_int = param_o.astype(int)
                ofile.write(', ')
                for i in range(cond):
                    c_out = np.sum((param_o[:,i] > 0).astype(int) + (param_o[:,1] > 0.9).astype(int)==2)
                    out_c = 'Outlier in %s = %d' %(c_label[i], c_out) 
                    output_info.append([run, out_c])
                    ofile.write('%s, ' %(c_out))
            else: 
                param_o = param
                for i in range(cond):
                    c_out = 0
                    out_c = 'Outlier in %s = %d' %(c_label[i], c_out) 
                    output_info.append([run, out_c])
            ofile.write('\n')
            # compute correlation coefficients
            stim_corr = []
            p_values = []
            #pa = param_o.astype(int)
            #pa2 = abs(pa)
            for i in range(cond):
                # correlate each motion parameter with each (i) condition onset
                mp1 = [scipy.stats.pearsonr(param_o[:,(i)], param_o[:,(cond)])]
                mp2 = [scipy.stats.pearsonr(param_o[:,(i)], param_o[:,(cond+1)])]
                mp3 = [scipy.stats.pearsonr(param_o[:,(i)], param_o[:,(cond+2)])]
                mp4 = [scipy.stats.pearsonr(param_o[:,(i)], param_o[:,(cond+3)])]
                mp5 = [scipy.stats.pearsonr(param_o[:,(i)], param_o[:,(cond+4)])]
                mp6 = [scipy.stats.pearsonr(param_o[:,(i)], param_o[:,(cond+5)])]
                # correlate sum of outliers with each (i) condition onset
                if outlier > 0:
                    out = [scipy.stats.pearsonr(param_o[:,(i)], param_o[:,1])]
                    stim_corr.append([[i,mp1[0][0]], [i, mp2[0][0]], [i, mp3[0][0]], [i, mp4[0][0]], [i, mp5[0][0]], [i, mp6[0][0]], [i, out[0][0]]])
                    p_values.append([[i,mp1[0][1]], [i, mp2[0][1]], [i, mp3[0][1]], [i, mp4[0][1]], [i, mp5[0][1]], [i, mp6[0][1]], [i, out[0][1]]])
                else:
                    stim_corr.append([[i,mp1[0][0]], [i, mp2[0][0]], [i, mp3[0][0]], [i, mp4[0][0]], [i, mp5[0][0]], [i, mp6[0][0]]])
                    p_values.append([[i,mp1[0][1]], [i, mp2[0][1]], [i, mp3[0][1]], [i, mp4[0][1]], [i, mp5[0][1]], [i, mp6[0][1]]])
            # save plot of parameter file (each run)
            max1 = np.amax(param_o)
            min1 = np.amin(param_o)
            fig1 = plt.figure(figsize=(12,6), dpi=80)
            fig1_title = plt.title("Parameter %s" %(run))
            # fig1_plot1 = plt.plot(param_o[:,0:(0+reg)], color='gray', label= r'$Regressor$')
            fig1_plot2 = plt.plot(param_o[:,(0):cond], color='blue', label=r'$Stimulus Onset$')
            fig1_plot3 = plt.plot(param_o[:,cond:(cond+6)], color='red', label=r'$Motion Parameter$')

            if outlier > 0:
                fig1_plot4 = plt.plot(param_o[:,columns], color='yellow', label=r'$Outlier Sum$')

            fig1_legend = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            fig1_ylim = plt.ylim(min10.5,max1+0.5)

            plt.savefig(os.path.abspath('parameter_img_%s.png' %(run)),bbox_extra_artists=(fig1_legend,), bbox_inches='tight')
            out_images1.append(os.path.abspath('parameter_img_%s.png'%run))
            
            # save image of pvalues for correlation coefficients (each run)
            p_values_fig = np.asarray(p_values)
            fig2 = plt.figure()
            fig2_title = plt.title("P Values %s" %(run))
            fig2_xticks = plt.xticks([0,1,2,3,4,5,6,7,8,10], c_label)
            if outlier > 0:
                fig2_yticks = plt.yticks([0,1,2,3,4,5,6], [r'$Motion1$', r'$Motion2$', r'$Motion3$', r'$Motion4$', r'$Motion5$', r'$Motion6$',  r'$OutlierSum$' ])
            else: 
                fig2_yticks = plt.yticks([0,1,2,3,4,5], [r'$Motion1$', r'$Motion2$', r'$Motion3$', r'$Motion4$', r'$Motion5$', r'$Motion6$'])
            ps = p_values_fig[:, :, 1]
            fig2_image = plt.imshow(ps.T, interpolation='nearest', cmap = plt.get_cmap('seismic_r'), vmin = 0, vmax = 0.1)
            cb = plt.colorbar()
            plt.savefig(os.path.abspath('p_values_img_%s.png' %(run)))
            out_images2.append(os.path.abspath('p_values_img_%s.png'%run))
        output1_path = os.path.abspath("output_check_%s.txt" %subject_id)
        np.savetxt(output1_path, np.asarray(output_info), fmt='%s')
        stim_path = os.path.abspath('stimulus_motion_correlation.csv')
        sfile = open(stim_path, 'w')
        sfile.write(', '.join(["Condition"]+["Motion%d" %d for d in range(6)] + ["Outliers"]))
        sfile.write('\n')
        for i, line in enumerate(stim_corr):
            print line
            sfile.write(', '.join([c_label[i]]+[str(l[1]) for l in line]))
            sfile.write('\n')
        sfile.close()
        p_path = os.path.abspath('p_values_correlation.csv')
        pfile = open(p_path,'w')
        pfile.write(', '.join(["Condition"]+["Motion %d" %d for d in range(6)]+["Outliers"]))
        pfile.write('\n') 
        for i,line in enumerate(p_values):
            print line
            pfile.write(', '.join([c_label[i]]+[str(l[1]) for l in line]))
            pfile.write('\n')
        pfile.close()
        ofile.close()
        return output_path, output1_path, out_images1, out_images2, stim_path, p_path
            
    if sparse:  
        for j, i in enumerate(subinfo):
            c_label = i.conditions
            cond = len(i.conditions)
            reg = len(i.regressor_names)
        output_path = os.path.abspath("Outlier.csv")
        ofile = open(output_path, 'w')
        ofile.write(', '.join(["Subject ID"]+["Run"]+["Outlier All"]+["Outlier in %s" %c_label[d] for d in range(cond)]))
        ofile.write('\n')        
        for r in range(len(subinfo)): 
            run = 'run%s' %(r)
            # path = os.path.join(inpath, '_generate_model%d/run%d.mat' %(r,r)) # 
            if range(len(subinfo)) > 0:
                param = np.genfromtxt(inpath[r], skip_header=5)
            else:
                param = np.genfromtxt(inpath, skip_header=5)
            mat = param.shape
            columns = param.shape[1]
            count = reg+6+cond
            outlier = columnscount
            out = 'Outlier = %d' %(outlier)
            regs = 'Regressors = %d' %(reg)
            con = 'Conditions = %d' %(cond)
            matr = 'Design Matrix Shape = [%d rows, %d columns]' %(mat)
            output_info.append([[run, out, regs, con, matr]])
            ofile.write(', '.join([str(subject_id)]+[str(r)]+[str(outlier)]))
            if outlier > 0:
                o = param[:, count:columns]
                o_sums = o.sum(axis=1)
                param_o = np.column_stack((param, o_sums))
                ofile.write(', ')
                for i in range(cond):
                    c_out = np.sum((param_o[:,i+reg+6] > 0).astype(int) + (param_o[:,1] > 0.9).astype(int)==2)
                    out_c = 'Outlier in %s = %d' %(c_label[i], c_out) 
                    output_info.append([run, out_c])
                    ofile.write('%s, ' %(c_out))
            else: 
                param_o = param
                c_out = 0
                ofile.write(', ')
                for i in range(cond):
                    out_c = 'Outlier in %s = %d' %(c_label[i], c_out) 
                    output_info.append([run, out_c])
                    ofile.write('%s, ' %(c_out))
            ofile.write('\n')

            # compute correlation coefficients
            stim_corr = []
            p_values = []
            for i in range(cond):
                # correlate each motion parameter with each (i) condition onset
                mp1 = [scipy.stats.pearsonr(param_o[:,(reg)], param_o[:,(i+reg+6)])]
                mp2 = [scipy.stats.pearsonr(param_o[:,(reg+1)], param_o[:,(i+reg+6)])]
                mp3 = [scipy.stats.pearsonr(param_o[:,(reg+2)], param_o[:,(i+reg+6)])]
                mp4 = [scipy.stats.pearsonr(param_o[:,(reg+3)], param_o[:,(i+reg+6)])]
                mp5 = [scipy.stats.pearsonr(param_o[:,(reg+4)], param_o[:,(i+reg+6)])]
                mp6 = [scipy.stats.pearsonr(param_o[:,(reg+5)], param_o[:,(i+reg+6)])]
                # correlate sum of outliers with each (i) condition onset
                if outlier > 0:
                    out = [scipy.stats.pearsonr(param_o[:,1], param_o[:,(i+reg+6)])]
                    stim_corr.append([[i,mp1[0][0]], [i, mp2[0][0]], [i, mp3[0][0]], [i, mp4[0][0]], [i, mp5[0][0]], [i, mp6[0][0]], [i, out[0][0]]])
                    p_values.append([[i,mp1[0][1]], [i, mp2[0][1]], [i, mp3[0][1]], [i, mp4[0][1]], [i, mp5[0][1]], [i, mp6[0][1]], [i, out[0][1]]])
                else:
                    stim_corr.append([[i,mp1[0][0]], [i, mp2[0][0]], [i, mp3[0][0]], [i, mp4[0][0]], [i, mp5[0][0]], [i, mp6[0][0]]])
                    p_values.append([[i,mp1[0][1]], [i, mp2[0][1]], [i, mp3[0][1]], [i, mp4[0][1]], [i, mp5[0][1]], [i, mp6[0][1]]])
            
            # save plot of parameter file (each run)
            max1 = np.amax(param_o)
            min1 = np.amin(param_o)
            fig1 = plt.figure(figsize=(12,6), dpi=80)
            fig1_title = plt.title("Parameter %s" %(run))
            fig1_plot1 = plt.plot(param_o[:,0:(0+reg)], color='gray', label= r'$Regressor$')
            fig1_plot2 = plt.plot(param_o[:,reg:(reg+6)], color='red', label=r'$Motion Parameter$')
            fig1_plot3 = plt.plot(param_o[:,(reg+6):count], color='blue', label=r'$Stimulus Onset$')
            if outlier > 0:
                fig1_plot4 = plt.plot(param_o[:,columns], color='yellow', label=r'$Outlier Sum$')

            fig1_legend = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            fig1_ylim = plt.ylim(min10.5,max1+0.5)

            plt.savefig(os.path.abspath('parameter_img_%s.png' %(run)),bbox_extra_artists=(fig1_legend,), bbox_inches='tight')
            out_images1.append(os.path.abspath('parameter_img_%s.png'%run))
            
            # save image of pvalues for correlation coefficients (each run)
            p_values_fig = np.asarray(p_values)
            fig2 = plt.figure()
            fig2_title = plt.title("P Values %s" %(run))
            fig2_xticks = plt.xticks([0,1,2,3,4,5,6,7,8,10], [r'$Cond1$', r'$Cond2$', r'$Cond3$', r'$Cond4$', r'$Cond5$', r'$Cond6$' ])
            if outlier > 0:
                fig2_yticks = plt.yticks([0,1,2,3,4,5,6], [r'$Motion1$', r'$Motion2$', r'$Motion3$', r'$Motion4$', r'$Motion5$', r'$Motion6$',  r'$OutlierSum$' ])
            else: 
                fig2_yticks = plt.yticks([0,1,2,3,4,5], [r'$Motion1$', r'$Motion2$', r'$Motion3$', r'$Motion4$', r'$Motion5$', r'$Motion6$'])
             ps = p_values_fig[:, :, 1]
            fig2_image = plt.imshow(ps.T, interpolation='nearest', cmap = plt.get_cmap('seismic_r'), vmin = 0, vmax = 0.1)
            cb = plt.colorbar()

            plt.savefig(os.path.abspath('p_values_img_%s.png' %(run)))
            out_images2.append(os.path.abspath('p_values_img_%s.png'%run))

        output1_path = os.path.abspath("output_check_%s.txt" %subject_id)
        np.savetxt(output1_path, np.asarray(output_info), fmt='%s')
        stim_path = os.path.abspath('stimulus_motion_correlation.csv')
        sfile = open(stim_path, 'w')
        sfile.write(', '.join(["Condition"]+["Motion%d" %d for d in range(6)] + ["Outliers"]))
        sfile.write('\n')
        for i, line in enumerate(stim_corr):
            print line
            sfile.write(', '.join([c_label[i]]+[str(l[1]) for l in line]))
            sfile.write('\n')
        sfile.close()
        p_path = os.path.abspath('p_values_correlation.csv')
        pfile = open(p_path,'w')
        pfile.write(', '.join(["Condition"]+["Motion%d"%d for d in range(6)]+["Outliers"]))
        pfile.write('\n') 
        for i,line in enumerate(p_values):
            print line
            pfile.write(', '.join([c_label[i]]+[str(l[1]) for l in line]))
            pfile.write('\n')
        pfile.close()
        ofile.close()
        return output_path, output1_path, out_images1, out_images2, stim_path, p_path

motionflow = pe.Workflow('stim_mot')
motionflow.base_dir=os.path.join(os.getcwd(),'stim_cor_working')
stim_mot = pe.Node(util.Function(input_names=['subinfo', 'inpath', 'sparse', 'subject_id'],
                                output_names=['output_path', 'output1_path', 'out_images1', 'out_images2', 'stim_path', 'p_path'],
                                function=stim_corr), name='stim_motion')
stim_mot.inputs.sparse = c.is_sparse
datagrabber = c.datagrabber.create_dataflow()
sink = pe.Node(nio.DataSink(), name='sink')
sink.inputs.base_directory = c.sink_dir
subjects = datagrabber.get_node('subject_id_iterable')
motionflow.connect(subjects,'subject_id',sink,'container')
subjectinfo = pe.Node(util.Function(input_names=['subject_id'], output_names=['output']), name='subjectinfo')
subjectinfo.inputs.function_str = c.subjectinfo
def getsubs(subject_id):#from config import getcontrasts, get_run_numbers, subjectinfo, fwhm
        
    subs = [('_subject_id_%s/'%subject_id,'')]

    return subs

get_substitutions = pe.Node(util.Function(input_names=['subject_id'],
    output_names=['subs'], function=getsubs), name='getsubs')

motionflow.connect(subjects,'subject_id',get_substitutions,'subject_id')
motionflow.connect(get_substitutions,"subs",sink,"substitutions")
motionflow.connect(datagrabber, 'datagrabber.input_files', stim_mot, 'inpath')
motionflow.connect(subjects,'subject_id',stim_mot,'subject_id')
motionflow.connect(subjectinfo,'output', stim_mot, 'subinfo')
motionflow.connect(subjects,'subject_id',subjectinfo,'subject_id')
motionflow.connect(stim_mot, 'output_path', sink, 'Stimulus_Motion.@file1')
motionflow.connect(stim_mot, 'output1_path', sink, 'Stimulus_Motion.@file2')
motionflow.connect(stim_mot,'out_images1',sink,'Stimulus_Motion.@images1')
motionflow.connect(stim_mot,'out_images2',sink,'Stimulus_Motion.@images2')
motionflow.connect(stim_mot,'stim_path',sink,'Stimulus_Motion.@parameter')
motionflow.connect(stim_mot,'p_path',sink,'Stimulus_Motion.@pvalues')
motionflow.run()


# In[8]:


#visualize conditions and hpf
hpf=120


# In[9]:


#dice coefficient between normed anat and template, and normed epi
normed_epi=
normed_anat=
masked_template=

dice = np.sum(seg[gt==k])*2.0 / (np.sum(seg) + np.sum(gt))

print 'Dice similarity score is {}'.format(dice)


# In[11]:


#visualize outliers (run art at several different levels)
def plot_timeseries(roi,statsfile,TR,plot,onsets,units):
    """ Returns a plot of an averaged timeseries across an roi
    
    Parameters
    ----------
    roi : List of ints
          List of integers corresponding to roi's in the Freesurfer LUT
    statsfile : File
                File output of segstats workflow
    TR : Float
         TR of scan
    plot : Boolean
           True to return plot
           
    Returns
    -------
    File : Filename of plot image, if plot=True 
    List : List of average ROI value if plot=False
    """
    import numpy as np
    import os
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    stats = np.recfromcsv(statsfile)     
    
    LUT = np.genfromtxt(os.path.join(os.environ["FREESURFER_HOME"],'FreeSurferColorLUT.txt'),dtype = str)
    roinum = LUT[:,0]
    roiname = LUT[:,1]
    Fname = []
    AvgRoi = []
    
    if roi == ['all']:
        roi = []
        for i, r in enumerate(stats):
            roi.append(list(r)[0]) 
    
    for R in roi:
        temp = False
        #ghetto for loop: find index of roi in stats list
        for i, r in enumerate(stats):
            if list(r)[0] == R:
                temp = True
                break    
        
        if temp:
            #find roi name for plot title
            title = roiname[roinum==str(np.int_(R))][0]
            if plot:
                nums = np.asarray(list(stats[i])[1:])
                X = np.array(range(len(nums)))*TR
                plt.figure(1)
                plt.plot(X,nums)
                if onsets:
                    # onsets is a Bunch with "conditions", "onsets" and "durations".
                    print onsets
                    names = onsets.conditions
                    durations = onsets.durations
                    onsets = onsets.onsets
                    colors1 = [[]]*len(onsets)

                    for i, ons in enumerate(onsets):
                        colors1[i] = [np.random.rand(3)]
                        if units == 'scans':
                            plt.plot(np.asarray(ons)*TR,nums[ons],marker='*',linestyle='None',color=colors1[i][0])
                        else:
                            plt.plot(ons,nums[np.int_(np.asarray(ons)/TR)],marker='*',linestyle='None',color=colors1[i][0])

                    plt.legend(['signal']+names)

                    for i, ons in enumerate(onsets):
                        ons = np.asarray(ons)
                        newX = np.zeros(nums.shape)
                        newX[:] = np.nan
                        for d in xrange(durations[i][0]):
                            if units == 'scans':
                                newX[np.int_(ons+np.ones(ons.shape)*(d))] = nums[np.int_(ons+np.ones(ons.shape)*(d))]
                            else:
                                newX[np.int_(ons/TR)] = nums[np.int_(ons/TR)]
                        plt.plot(X,newX,color=colors1[i][0])


                plt.title(title)
                plt.xlabel('time (s)')
                plt.ylabel('signal')

                fname = os.path.join(os.getcwd(),os.path.split(statsfile)[1][:-4]+'_'+title+'.png')
                plt.savefig(fname,dpi=200)
                plt.close()
                Fname.append(fname)
            else:
                AvgRoi.append([title,np.mean(list(stats[i])[1])])
        else:
            print "roi %s not found!"%R

    return Fname, AvgRoi
    
    def plot_ADnorm(ADnorm,TR,norm_thresh,out):
    """ Returns a plot of the composite_norm file output from art
    
    Parameters
    ----------
    ADnorm : File
             Text file output from art
    TR : Float
         TR of scan
         
    Returns
    -------
    File : Filename of plot image
    
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import os
    import numpy as np

    if not isinstance(out,list):
        out = [out]

    plot = os.path.abspath('plot_'+os.path.split(ADnorm)[1]+'.png')
    
    data = np.genfromtxt(ADnorm)
    plt.figure(1,figsize = (8,3))
    X = np.array(range(data.shape[0]))*TR
    plt.plot(X,data)
    plt.xlabel('Time (s)')
    plt.ylabel('Composite Norm')
    
    if norm_thresh > max(data):
        plt.axis([0,TR*data.shape[0],0,norm_thresh*1.1])
        plt.plot(X,np.ones(X.shape)*norm_thresh)
        for o in out:
            plt.plot(o*TR*np.ones(2),[0,norm_thresh*1.1],'r-')
    else:
        plt.axis([0,TR*data.shape[0],0,max(data)*1.1])
        plt.plot(X,np.ones(X.shape)*norm_thresh)
        for o in out:
            plt.plot(o*TR*np.ones(2),[0,max(data)*1.1],'r-')
    
    plt.savefig(plot,bbox_inches='tight')
    plt.close()
    return plot
    
    def art_output(art_file,intensity_file,stats_file):
    import numpy as np
    from nipype.utils.filemanip import load_json
    import os
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    try:
        out=np.asarray(np.genfromtxt(art_file))
    except:
        out=np.asarray([])
    table=[["file",art_file],["num outliers", str(out.shape)],["timepoints",str(out)]]
    stats = load_json(stats_file)
    for s in stats:
        for key, item in s.iteritems():
            if isinstance(item,dict):
                table.append(['+'+key,''])
                for sub_key,sub_item in item.iteritems():
                    table.append(['  '+sub_key,str(sub_item)])
            elif isinstance(item, list):
                table.append(['+'+key,''])
                for s_item in item:
                    for sub_key, sub_item in s_item.iteritems():
                        table.append(['  '+sub_key,str(sub_item)])
            else:
                table.append([key,str(item)])
    print table
    intensity = np.genfromtxt(intensity_file)
    intensity_plot = os.path.abspath('global_intensity.png')
    plt.figure(1,figsize = (8,3))
    plt.xlabel('Volume')
    plt.ylabel("Global Intensity")
    plt.plot(intensity)
    plt.savefig(intensity_plot,bbox_inches='tight')
    plt.close()
    return table, out.tolist(), intensity_plot


# In[ ]:


#plot meanfunc/anat in func space
#plot mni mean func/ mni normed anat/ mni template

