#!/usr/bin/python

import numpy as np
import mvpa2.suite as mv
from glob import glob
from sklearn.cross_validation import LeaveOneOut, StratifiedKFold
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support
from sklearn.metrics import cohen_kappa_score, confusion_matrix, recall_score
from sklearn.grid_search import GridSearchCV
from nilearn.signal import clean


base_dir = '/data/movieloc/backup_store/saccs/'
data_dir = '/func/'
anat_dir = '/anat/'
template_dir = '/xfm/'
figures_dir = base_dir + 'figures/'

participants = sorted([path.split('/')[-1] for path in glob(base_dir + 'sub-*')])
rois = ['FFA', 'OFA', 'PPA', 'EBA', 'LOC', 'VIS']

# Load masked movie data IN GROUP TEMPLATE SPACE and assign ROIs as feature attributes

# Set order of polynomial for detrending
polyord = 3

movie_dss = []
for participant in participants:
    # Load movie data with brain mask for a participant
    movie_fns = sorted(glob(base_dir + participant + data_dir + '*_task-avmovie_run-*highpass_tmpl.nii.gz'))
    mask_fn = base_dir + participant + anat_dir + 'brain_mask_tmpl.nii.gz'
    assert len(movie_fns) == 8
    
    # Include chunk (i.e., run) labels
    movie_ds = mv.vstack([mv.fmri_dataset(movie_fn, mask=mask_fn, chunks=run)
                          for run, movie_fn in enumerate(movie_fns)])
    
    # Assign participant labels as feature attribute
    movie_ds.fa['participant'] = [participant] * movie_ds.shape[1]
    print("Loaded movie data for participant {0}".format(participant))
    
    # Perform linear detrending per chunk
    mv.poly_detrend(movie_ds, polyord=polyord, chunks_attr='chunks')
    
    # Perform low-pass filtering per chunk
    movie_ds.samples = clean(movie_ds.samples, sessions=movie_ds.sa.chunks, low_pass=.1,
                             high_pass=None, t_r=2.0, detrend=False, standardize=False)

    # Z-score movie time series per chunk
    mv.zscore(movie_ds, chunks_attr='chunks')
    print("Finished preprocessing (detrending, z-scoring) for participant {0}".format(participant))

    # Load ROI masks and attach them to movie data
    all_rois_mask = np.array([['brain'] * movie_ds.shape[1]]).astype('S10')
    for roi in rois:
        # Get filenames for potential right and left ROI masks
        if roi == 'VIS':
            roi_fns = sorted(glob(base_dir + participant + anat_dir +
                         '{0}_*_mask_tmpl.nii.gz'.format(roi)))
        else:
            left_roi_fns = sorted(glob(base_dir + participant + anat_dir + 
                             'l{0}_*_mask_tmpl.nii.gz'.format(roi)))
            right_roi_fns = sorted(glob(base_dir + participant + anat_dir +
                             'r{0}_*_mask_tmpl.nii.gz'.format(roi)))
            roi_fns = left_roi_fns + right_roi_fns

        if len(roi_fns) == 0:
            print("ROI {0} does not exist for participant {1}; appending all zeros".format(roi, participant))
            roi_mask = np.zeros((1, movie_ds.shape[1]))

        elif len(roi_fns) == 1:
            roi_mask = mv.fmri_dataset(roi_fns[0], mask=mask_fn).samples


        elif len(roi_fns) > 1:
            
            # Add ROI maps into single map
            print("Combining {0} {1} masks for participant {2}".format(
                      len(roi_fns), roi, participant))
            roi_mask = np.sum([mv.fmri_dataset(roi_fn, mask=mask_fn).samples for roi_fn in roi_fns],
                            axis=0)
            
            # Set any voxels that might exceed 1 to 1
            roi_mask = np.where(roi_mask > 0, 1, 0)

        # Ensure that number of voxels in ROI mask matches movie data
        assert roi_mask.shape[1] == movie_ds.shape[1]
        
        # Flatten mask into list
        roi_flat = list(roi_mask.ravel())

        # Assign ROI mask to movie data feature attributes
        movie_ds.fa[roi] = roi_flat

        # Get lateralized masks as well
        if roi != 'VIS':
            lat_roi_mask = np.zeros((1, movie_ds.shape[1]))
            if len(left_roi_fns) == 1:
                left_roi_mask = np.where(mv.fmri_dataset(left_roi_fns[0],
                                                      mask=mask_fn).samples > 0, 1, 0)
                lat_roi_mask[left_roi_mask > 0] = 1
            elif len(left_roi_fns) > 1:
                left_roi_mask = np.where(np.sum([mv.fmri_dataset(left_roi_fn,
                                                      mask=mask_fn).samples for 
                                              left_roi_fn in left_roi_fns], axis=0) > 0, 1, 0)
                lat_roi_mask[left_roi_mask > 0] = 1
            elif len(left_roi_fns) == 0:
                left_roi_mask = np.zeros((1, movie_ds.shape[1]))
            
            if len(right_roi_fns) == 1:
                right_roi_mask = np.where(mv.fmri_dataset(right_roi_fns[0],
                                                      mask=mask_fn).samples > 0, 1, 0)
                lat_roi_mask[right_roi_mask > 0] = 2
            elif len(right_roi_fns) > 1:
                right_roi_mask = np.where(np.sum([mv.fmri_dataset(right_roi_fn,
                                                      mask=mask_fn).samples for 
                                              right_roi_fn in right_roi_fns], axis=0) > 0, 1, 0)
                lat_roi_mask[right_roi_mask > 0] = 2
            elif len(right_roi_fns) == 0:
                right_roi_mask = np.zeros((1, movie_ds.shape[1]))
                


            # Ensure that number of voxels in ROI mask matches movie data
            assert lat_roi_mask.shape[1] == movie_ds.shape[1]

            # Flatten mask into list
            lat_roi_flat = list(lat_roi_mask.ravel())

            # Assign ROI mask to movie data feature attributes
            movie_ds.fa['lat_' + roi] = lat_roi_flat
            
            # Check existing feature attribute for all ROIS for overlaps
            np.place(all_rois_mask, ((left_roi_mask > 0) | (right_roi_mask > 0))
                     & (all_rois_mask != 'brain'), 'overlap')

            all_rois_mask[(left_roi_mask > 0) & (all_rois_mask != 'overlap')] = 'left {0}'.format(roi)
            all_rois_mask[(right_roi_mask > 0) & (all_rois_mask != 'overlap')] = 'right {0}'.format(roi)
            

        elif roi == 'VIS':
            roi_fns = sorted(glob(base_dir + participant + anat_dir +
                            '{0}_*_mask_tmpl.nii.gz'.format(roi)))
            roi_mask = np.sum([mv.fmri_dataset(roi_fn, mask=mask_fn).samples for roi_fn in roi_fns],
                              axis=0)
            np.place(all_rois_mask, (roi_mask > 0) & (all_rois_mask != 'brain'), 'overlap')
            all_rois_mask[(roi_mask > 0) & (all_rois_mask != 'overlap')] = roi
            

    # Flatten mask into list
    all_rois_flat = list(all_rois_mask.ravel())

    # Assign ROI mask to movie data feature attributes
    movie_ds.fa['all_ROIs'] = all_rois_flat        
        
    movie_dss.append(movie_ds)

    mv.h5save(base_dir + participant + data_dir + '{0}_avmovie_detrend{1}_lowpass_ROIs_tmpl_bold.hdf5'.format(participant, polyord), movie_ds)
    print("Finished participant {0}, saved the data".format(participant))

mv.h5save(base_dir + 'allsub_avmovie_detrend{0}_lowpass_ROIs_tmpl_bold.hdf5'.format(polyord), movie_dss)

# Horizontally stack all data sets
ds_wide = mv.hstack(movie_dss)



# Transpose brain so voxels are now samples
ds = mv.Dataset(ds_wide.samples.T, sa=ds_wide.fa.copy(), fa=ds_wide.sa.copy())


# Save transposed data
mv.h5save(base_dir + 'allsub_transpose_avmovie_detrend{0}_lowpass_ROIs_tmpl_bold.hdf5'.format(polyord), ds)



