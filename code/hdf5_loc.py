#!/home/adina/wtf/bin/python
# i think shebang was the culprit
# # !/usr/bin/python


import numpy as np
import mvpa2.suite as mv
from glob import glob
from sklearn.cross_validation import (LeaveOneOut,
                                    StratifiedKFold)
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (accuracy_score,
                            roc_auc_score,
                            precision_recall_fscore_support)
from sklearn.metrics import (cohen_kappa_score,
                            confusion_matrix,
                            recall_score)
from sklearn.grid_search import GridSearchCV
#from nilearn.signal import clean
import itertools

basedir='/data/movieloc/backup_store/saccs/'
locdir='/ses-localizer/'
anat='/anat/'
"""
This script performs Yariks and Adinas "approach 1". It computes a simple GLM in
which sensitivities of a particular linear decision between two brain areas are
regressed onto the events files of the underlying block design. Data basis is
the studyforrest phase 2 localizer session. The motion corrected and "aligned"
datafiles were preprocessed (smoothing, brain extraction, high-pass filtering).

################################################################################
############################### OBJECTIVE ######################################

- per subject, load the data into an hdf5 file. Input options for rois,
  subject. NO polynomial detrending, but zscoring
- create and train GNB classifier on data.
- compute sensitivities with sensitivity analyser
- save sensitivities in subject folder as sens.hdf5 something with h5save.
  make sure the resulting dataset with sensitivities contains chunks (timepoints?)
- get the event files from the aligned dataset. create detailed event files
  that also include an 'category-begins' event

"""
########
# step 1:
# lets get the data, and build the full dataset
def buildthisshit():
    """buildthisshit() will build and save participant-specific hdf5 datasets
    with all rois from preprocessed objectcategories data, stack them for a
    group dataset and save them, and transpose the group dataset and save it."""

    participants = sorted([path.split('/')[-1] for path in glob(basedir + 'sub-*')])
    rois = ['FFA', 'OFA', 'PPA', 'EBA', 'LOC', 'VIS']
    movie_dss = []

    for participant in participants:
        movie_fns = sorted(glob(basedir + participant + locdir + '/func/' + \
                                '{}_task-objectcategories_run-*_space-custom-subject_desc-highpass_bold.nii.gz'.format(participant)))

        mask_fn = basedir + participant + anat + 'brain_mask.nii.gz'
        assert len(movie_fns)==4

        movie_ds = mv.vstack([mv.fmri_dataset(movie_fn, mask=mask_fn, chunks=run)
                                for run, movie_fn in enumerate(movie_fns)])

        movie_ds.fa['participant']=[participant] * movie_ds.shape[1]
        print('loaded movie data for participant {}.'.format(participant))

        #zscore stuff
        mv.zscore(movie_ds, chunks_attr='chunks')
        print('finished zscoring for participant {}.'.format(participant))

        all_rois_mask = np.array([['brain']*movie_ds.shape[1]]).astype('S10')
        for roi in rois:
            # Get filenames for potential right and left ROI masks
            if roi == 'VIS':
                roi_fns = sorted(glob(basedir + participant + anat + \
                                        '{0}_*_mask.nii.gz'.format(roi)))
            else:
                left_roi_fns = sorted(glob(basedir + participant + anat + \
                                            'l{0}_*_mask.nii.gz'.format(roi)))
                right_roi_fns = sorted(glob(basedir + participant + anat + \
                                            'r{0}_*_mask.nii.gz'.format(roi)))
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
                roi_mask = np.sum([mv.fmri_dataset(roi_fn, mask=mask_fn).samples for roi_fn in roi_fns], axis=0)

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
                roi_fns = sorted(glob(basedir + participant + anat + '/{0}_*_mask.nii.gz'.format(roi)))
                roi_mask = np.sum([mv.fmri_dataset(roi_fn, mask=mask_fn).samples for roi_fn in roi_fns],
                                  axis=0)
                np.place(all_rois_mask, (roi_mask > 0) & (all_rois_mask != 'brain'), 'overlap')
                all_rois_mask[(roi_mask > 0) & (all_rois_mask != 'overlap')] = roi
        # Flatten mask into list
        all_rois_flat = list(all_rois_mask.ravel())

        # Assign ROI mask to movie data feature attributes
        movie_ds.fa['all_ROIs'] = all_rois_flat
        # saving individual data
        mv.h5save(basedir + participant + locdir + '/func/' + \
        '{}_ses-localizer_task-objectcategories_ROIs_space-custom-subject_desc-highpass.hdf5'.format(participant),
        movie_ds)

        # join all datasets
        movie_dss.append(movie_ds)

    # save full dataset
    mv.h5save(basedir + 'ses-localizer_task-objectcategories_ROIs_space-custom-subject_desc-highpass.hdf5', movie_dss)
    print('saved the collection of all subjects datasets as {}.'.format(basedir +
    'ses-localizer_task-objectcategories_ROIs_space-custom-subject_desc-highpass.hdf5'))

    # squish everything together
    ds_wide = mv.hstack(movie_dss)

    # transpose the dataset, time points are now features
    ds = mv.Dataset(ds_wide.samples.T, sa=ds_wide.fa.copy(), fa=ds_wide.sa.copy())
    mv.h5save(basedir + 'ses-localizer_task-objectcategories_ROIs_space-custom-subject_desc-highpass_transposed.hdf5', ds)
    print('Transposed the group-dataset and saved it as {}.'.format(basedir +
    'ses-localizer_task-objectcategories_ROIs_space-custom-subject_desc-highpass_transposed.hdf5'))
    return ds


def dothefuckingclassification(ds):
    """ Dothefuckingclassification does the fucking classification. It builds a
    linear gaussian naive bayes classifier, performs a leave-one-out
    crossvalidation and stores the sensitivities from the GNB classifier of each
    fold in a combined dataset for further use in a glm."""
    # set up classifier
    prior='ratio'
    targets= 'all_ROIs'
    gnb = mv.GNB(common_variance=True, prior=prior, space=targets)

    # prepare for callback of sensitivity extraction within CrossValidation
    sensitivities=[]
    def store_sens(data, node, result):
        sens = node.measure.get_sensitivity_analyzer(force_train=False)(data)
        # we also need to manually append the time attributes to the sens ds
        sens.fa['time_coords']=data.fa['time_coords']
        sens.fa['chunks']=data.fa['chunks']
        print('Storing sensitivity for data on participants %s' %
            str(data.sa['participant'].unique))
        sensitivities.append(sens)

    # do a crossvalidation classification
    cv = mv.CrossValidation(gnb, mv.NFoldPartitioner(attr='participant'),
                            errorfx=mv.mean_match_accuracy,
                            enable_ca=['stats'],
                            callback=store_sens)
    results = cv(ds)
    # save classification results
#    with open('/data/movieloc/backup_store/gnb_cv_results/objectcategory_clf.txt', 'a') as f:
#        f.write(cv.ca.stats.as_string(description=True))
    #save the results dataset
    mv.h5save(basedir + 'gnb_cv_classification_results.hdf5', results)
    print('Saved the crossvalidation results at {}.'.format(basedir +
    'gnb_cv_classification_results.hdf5'))
    mv.h5save(basedir + 'sensitivities_nfold.hdf5', sensitivities)
    print('Saved the sensitivities at {}.'.format(basedir +
    'sensitivities_nfold.hdf5'))
    # results now has the overall accuracy. results.samples gives the
    # accuracy per participant.
    # sensitivities contains a dataset for each participant with the
    # sensitivities as samples and class-pairings as attributes
    return sensitivities

    ########
    # step 3:
    # get event files. they're located in sourcedata/phase2/sub-*/ses-localizer/func/sub*_ses-localizer_task-objectcategories_run*_events.tsv
    # TODO: move event files into subs ses-localizer dir

def dothefuckingglm(sensitivities):    ## CODE TO CALCULATE ONE GLM PER SUBJECT?
    """dothefuckingglm does the fucking glm. It will squish the sensitivity
    dataset by vstacking them, calculating the mean sensitivity per ROI pair
    with the mean_group_sample() function, transpose it with a
    TransposeMapper(). It will get the event files and read them in, average the
    durations because there are tiny differences between subjects, and then it
    will put all of that into a glm.
    """
    sensitivities_stacked = mv.vstack(sensitivities)
    sensitivities_stacked.sa['all_ROIs_str'] = map(lambda p: '_'.join(p),
                                            sensitivities_stacked.sa.all_ROIs)
    mean_sens = mv.mean_group_sample(['all_ROIs_str'])(sensitivities_stacked)
    mean_sens_transposed = mean_sens.get_mapped(mv.TransposeMapper())
    # average onsets into one event file
    event_files = sorted(glob(basedir + '/sourcedata/phase2/*/ses-localizer/func/sub-*_ses-localizer_task-objectcategories_run-*_events.tsv'))
    vals = None
    for idx, filename in enumerate(event_files, 1):
        data = np.genfromtxt(filename, dtype=None, delimiter='\t', skip_header=1,
        usecols=(0,))
        if vals is None:
            vals = data
        else:
            vals += data
    meanvals = vals / idx
    events = np.genfromtxt(filename, delimiter='\t', names=True, dtype=[('onset',float),('duration', float),('trial_type', '|S16'), ('stim_file', '|S60')])
    for row, val in itertools.izip(events, meanvals):
        row['onset']=val
    for filename in event_files:
        d = np.genfromtxt(filename, delimiter='\t', names=True, dtype=[('onset',float), ('duration', float),('trial_type', '|S16'), ('stim_file', '|S60')])
        for i in range(0, len(d)):
            import numpy.testing as npt
            npt.assert_almost_equal(events['onset'][i], d['onset'][i], decimal = 0)
            npt.assert_almost_equal(events['duration'][i], d['duration'][i], decimal = 0)
            assert events['trial_type'][i] == d['trial_type'][i]

    # account for more variance by coding the first occurance
    i = 1
    while i < len(events):
        if i == 1:
            events[i-1]['trial_type'] = events[i-1]['trial_type'] + '_first'
            i += 1
        if events[i-1]['trial_type'] != events[i]['trial_type']:
            events[i]['trial_type'] = events[i]['trial_type'] + '_first'
            i += 2
        else:
            i += 1
    # save the event_file
    fmt = "%10.3f\t%10.3f\t%16s\t%60s"
    np.savetxt(basedir + 'group_events.tsv', events, delimiter='\t', comments='', \
     header='onset\tduration\ttrial_type\tstim_file', fmt=fmt)
    print('Combined all event files into a group event file at {}.'.format(basedir + 'group_events.tsv'))
    # get events into dictionary
    dicts = []
    for i in range(0, len(events)):
        dic = {'onset':events[i][0], 'duration':events[i][1], 'condition':events[i][2]}
        dicts.append(dic)

    hrf_estimates = mv.fit_event_hrf_model(mean_sens_transposed,
                                            dicts,
                                            time_attr='time_coords',
                                            condition_attr='condition',
                                            design_kwargs=dict(drift_model='blank'),
                                            glmfit_kwargs=dict(model='ols'),
                                            return_model=True)
    mv.h5save(basedir + 'sens_glm_objectcategories_results.hdf5', hrf_estimates)
    print('calculated glm, saving results at {}.'.format(basedir +
    'sens_glm_objectcategories_results.hdf5'))
    print('I am done with this bloody glm')
    return hrf_estimates


if __name__ == '__main__':
    import os.path
    # check whether data for subjects exists already, so that we can skip
    # buildthisshit()
    groupdata= basedir + 'ses-localizer_task-objectcategories_ROIs_space-custom-subject_desc-highpass_transposed.hdf5'
    sensdata = basedir + 'sensitivities_nfold.hdf5'
    ev_file = basedir + 'group_events.tsv'
    if os.path.isfile(groupdata):
        ds = mv.h5load(groupdata)
	print('loaded already existing transposed group data')
    else:
        ds = buildthisshit()

    if (os.path.isfile(sensdata)) and (os.path.isfile(ev_file)):
        sensitivities = mv.h5load(sensdata)
	print('loaded already existing sensitivities')
        events = np.genfromtxt(ev_file, names=('onset', 'duration',
        'trial_type', 'stim_file'), dtype=['<f8', '<f8', '|S18', '|S60'], skip_header=1)
	print('loaded already existing group eventfile')
    else:
 	sensitivities = dothefuckingclassification(ds)
    hrf_estimates = dothefuckingglm(sensitivities)

