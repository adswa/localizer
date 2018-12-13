#!/home/adina/wtf/bin/python

#--> mess on hydra currently requires python to be executed from virtual env
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
from nilearn.signal import clean
import itertools

basedir='/data/movieloc/backup_store/saccs/'
locdir='/ses-localizer/'
anat='/anat/'


## TODO: this should be taken care of
##reorder_full =[1, 0, 4, 9, 5, 10, 3, 8, 2, 7, 6, 11] # without overlap
##reorder_roi_only = [0, 3, 8, 4, 9, 2, 7, 1, 6, 5, 10]


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

### TODOS: introduce arguments to set up all of the analysis appropriately
### get rid of "overlap" category


######################################
#
# Some helper functions for later use.
#
######################################
# to strip a dataset from 'brain' and 'overlap'
def strip_ds(ds):
    """this helper takes a dataset with brain and overlap ROIs and strips these
    categories from it."""
    if 'brain' in np.unique(ds.sa.all_ROIs):
        ds = ds[(ds.sa.all_ROIs != 'brain'), :]
        print('excluded the rest of the brain from the dataset')
    if 'overlap' in np.unique(ds.sa.all_ROIs):
        ds = ds[(ds.sa.all_ROIs != 'overlap'), :]
        print('excluded overlap from the dataset')
    return ds


# to reorder labels for plotting
def get_known_labels(desired_order, known_labels):
    """ Start of a helper function to reorder ROI labels in a confusion matrix."""
    return [
        label
        for label in desired_order
        if label in known_labels
    ]

# to get a dataset with coordinates
def get_voxel_coords(ds, append=True, zscore=True):
    ds_coords = ds.copy('deep')
    # Append voxel coordinates (and squares, cubes)
    products = np.column_stack((ds.sa.voxel_indices[:, 0]*ds.sa.voxel_indices[:, 1],
                                ds.sa.voxel_indices[:, 0]*ds.sa.voxel_indices[:, 2],
                                ds.sa.voxel_indices[:, 1]*ds.sa.voxel_indices[:, 2],
                                ds.sa.voxel_indices[:, 0]*ds.sa.voxel_indices[:, 1]*ds.sa.voxel_indices[:, 2]))
    coords = np.hstack((ds.sa.voxel_indices,
                        ds.sa.voxel_indices**2,
                        ds.sa.voxel_indices**3,
                        products))
    coords = mv.Dataset(coords, sa=ds_coords.sa)
    if zscore:
        mv.zscore(coords, chunks_attr='participant')
    ds_coords.fa.clear()
    if append:
        ds_coords.samples = np.hstack((ds_coords.samples, coords.samples))
    elif not append:
        ds_coords.samples = coords.samples
    return ds_coords


# to get a group event file
def get_group_events():
    event_files = sorted(glob(basedir + '/sourcedata/phase2/*/ses-localizer/func/sub-*_ses-localizer_task-objectcategories_run-*_events.tsv'))
    vals = None
    for idx, filename in enumerate(event_files, 1):
        data = np.genfromtxt(filename,
                            dtype=None,
                            delimiter='\t',
                            skip_header=1,
                            usecols=(0,))
        if vals is None:
            vals = data
        else:
            vals += data
    meanvals = vals / idx
    events = np.genfromtxt(filename,
                            delimiter='\t',
                            names=True,
                            dtype=[('onset',float),('duration', float),('trial_type', '|S16'), ('stim_file', '|S60')])
    for row, val in itertools.izip(events, meanvals):
        row['onset']=val
    for filename in event_files:
        d = np.genfromtxt(filename,
                        delimiter='\t',
                        names=True,
                        dtype=[('onset',float), ('duration', float),('trial_type', '|S16'), ('stim_file', '|S60')])
        for i in range(0, len(d)):
            import numpy.testing as npt
            npt.assert_almost_equal(events['onset'][i], d['onset'][i], decimal=0)
            npt.assert_almost_equal(events['duration'][i], d['duration'][i], decimal=0)
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
    return events


# to get a baseline activation for zscoring
def extract_baseline(events, localizer_ds):
    """ This function extracts the mean and standard deviation for z-scoring
    from those times of the experiments with no stimulation. It is meant to be
    used within building of the group dataset."""
    # get last stimulation before break
    import bisect

    rest_periods_start = []
    rest_periods_end = []
    for i in range(len(events)-1):
        if i == 0:
            rest_periods_start.append(0.0)
            rest_periods_end.append(events[i]['onset'])
        else:
            dur = events[i+1]['onset'] - events[i]['onset']
            if dur > 5.0:
                rest_periods_start.append(events[i]['onset']+events[i]['duration'])
                rest_periods_end.append(events[i+1]['onset'])
    # append the last stimulation end, and the end of the scan
    rest_periods_start.append(events[-1]['onset']+events[-1]['duration'])
    rest_periods_end.append(localizer_ds.sa.time_coords[-1])
    assert len(rest_periods_start) == len(rest_periods_end)
    # extract the activation within the time slots and compute mean and std.
    # a bit tricky as event file and activation data exist in different time
    # resolutions. Will settle for an approximate solution, where I extract the
    # time coordinate closest to the onset and offset of restperiods
    restdata = []
    for i in range(len(rest_periods_start)):
        start_idx = bisect.bisect_left(localizer_ds[localizer_ds.sa.chunks ==
        0].sa.time_coords, rest_periods_start[i])
        end_idx = bisect.bisect_left(localizer_ds[localizer_ds.sa.chunks ==
        0].sa.time_coords, rest_periods_end[i])
        restdata.append(localizer_ds.samples[start_idx:end_idx])
    # flatten the list of arrays
    rests = np.concatenate(restdata)
    # get mean and std as per-feature vectors
    means = np.mean(rests, axis=0)
    std = np.std(rests, axis=0)
    return means, std


# to print  a confusion matrix
def plot_confusion(cv,
                   labels,
                   fn=None,
                   figsize=(9, 9),
                   ACC=None,
                   vmax=None):
    """ This function plots the classification results as a confusion matrix.
    Specify ACC as cv.ca.stats.stats['mean(ACC)'] to display accuracy in the
    title. Set a new upper boundery of the scale with vmax. To save the plot,
    specify a path/with/filename.png as the fn parameter. """

    import seaborn as sns
    import matplotlib.pyplot as plt
    origlabels = cv.ca.stats.labels
    origlabels_indexes = dict([(x,i) for i,x in enumerate(origlabels)])
    reorder = [origlabels_indexes.get(labels[i]) for i in range(len(labels))]
    matrix = cv.ca.stats.matrix[reorder][:, reorder].T
    # Plot matrix with color scaled to 90th percentile
    fig, ax = plt.subplots(figsize=figsize)
    im = sns.heatmap(100*matrix.astype(float)/np.sum(matrix, axis=1)[:, None],
                     cmap='gist_heat_r',
                     annot=matrix,
                     annot_kws={'size': 8},
                     fmt=',',
                     square=True,
                     ax=ax,
                     vmin=0,
                     vmax=vmax or np.percentile(matrix, 90),
                     xticklabels=labels,
                     yticklabels=labels)
    ax.xaxis.tick_top()
    if ACC:
        plt.suptitle('Mean accuracy of classification: {}'.format(ACC))
    plt.xticks(rotation=90)
    plt.xlabel('Predicted labels')
    plt.ylabel('Actual labels')
    ax.xaxis.set_label_position('top')
    plt.tight_layout()
    if fn:
        plt.savefig(fn)
    else:
    # if matrix isn't saved, just show it
        plt.show()



#########################################
#
# Dataset creation, classification, GLM.
#
#########################################

#######
# step 1:
# lets get the data, and build the full dataset
def buildadataset(zscore):
    """buildataset() will build and save participant-specific hdf5 datasets
    with all rois from preprocessed objectcategories data, stack them for a
    group dataset and save them, and transpose the group dataset and save it.
    The parameter 'zscore' determines whether and what kind of z-scoring
    should be performed."""
    print('I am building a dataset with the following option: {}.'.format(zscore))

    # get the participants and rois
    participants = sorted([path.split('/')[-1] for path in glob(basedir + 'sub-*')])
    rois = ['FFA', 'OFA', 'PPA', 'EBA', 'LOC', 'VIS']
    localizer_dss = []

    for participant in participants:
        localizer_fns = sorted(glob(basedir + participant + locdir + '/func/' + \
                                '{}_task-objectcategories_run-*_space-custom-subject_desc-highpass_bold.nii.gz'.format(participant)))
        mask_fn = basedir + participant + anat + 'brain_mask.nii.gz'
        assert len(localizer_fns)==4
        localizer_ds = mv.vstack([mv.fmri_dataset(localizer_fn, mask=mask_fn, chunks=run)
                                for run, localizer_fn in enumerate(localizer_fns)])

        localizer_ds.fa['participant']=[participant] * localizer_ds.shape[1]
        print('loaded localizer data for participant {}.'.format(participant))

        # zscore the data with means and standard deviations from no-stimulation
        # periods
        if zscore == 'custom':
            events = get_group_events()
            means, stds = extract_baseline(events, localizer_ds)
        #zscore stuff
            mv.zscore(localizer_ds, params = (means, stds), chunks_attr='chunks')
            print('finished custom zscoring for participant {}.'.format(participant))
        elif zscore == 'z-score':
            mv.zscore(localizer_ds, chunks_attr='chunks')
            print('finished zscoring for participant {}.'.format(participant))
        else:
            print('I did not zscore.')

        all_rois_mask = np.array([['brain']*localizer_ds.shape[1]]).astype('S10')
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
                roi_mask = np.zeros((1, localizer_ds.shape[1]))
            elif len(roi_fns) == 1:
                roi_mask = mv.fmri_dataset(roi_fns[0], mask=mask_fn).samples
            elif len(roi_fns) > 1:
                # Add ROI maps into single map
                print("Combining {0} {1} masks for participant {2}".format(
                          len(roi_fns), roi, participant))
                roi_mask = np.sum([mv.fmri_dataset(roi_fn, mask=mask_fn).samples for roi_fn in roi_fns], axis=0)
                # Set any voxels that might exceed 1 to 1
                roi_mask = np.where(roi_mask > 0, 1, 0)
            # Ensure that number of voxels in ROI mask matches localizer data
            assert roi_mask.shape[1] == localizer_ds.shape[1]
            # Flatten mask into list
            roi_flat = list(roi_mask.ravel())
            # Assign ROI mask to localizer data feature attributes
            localizer_ds.fa[roi] = roi_flat
            # Get lateralized masks as well
            if roi != 'VIS':
                lat_roi_mask = np.zeros((1, localizer_ds.shape[1]))
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
                    left_roi_mask = np.zeros((1, localizer_ds.shape[1]))

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
                    right_roi_mask = np.zeros((1, localizer_ds.shape[1]))

                # Ensure that number of voxels in ROI mask matches localizer data
                assert lat_roi_mask.shape[1] == localizer_ds.shape[1]
                # Flatten mask into list
                lat_roi_flat = list(lat_roi_mask.ravel())
                # Assign ROI mask to localizer data feature attributes
                localizer_ds.fa['lat_' + roi] = lat_roi_flat
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
        # Assign ROI mask to localizer data feature attributes
        localizer_ds.fa['all_ROIs'] = all_rois_flat
        # saving individual data TODO: for now in results dir!
        mv.h5save(results_dir + participant + locdir + '/func/' + \
        '{}_ses-localizer_task-objectcategories_ROIs_space-custom-subject_desc-highpass.hdf5'.format(participant),
        localizer_ds)
        print('Saved dataset for {}.'.format(participant))
        # join all datasets
        localizer_dss.append(localizer_ds)

    # save full dataset
    mv.h5save(results_dir +'ses-localizer_task-objectcategories_ROIs_space-custom-subject_desc-highpass.hdf5', localizer_dss)
    print('saved the collection of all subjects datasets.')
    # squish everything together
    ds_wide = mv.hstack(localizer_dss)

    # transpose the dataset, time points are now features
    ds = mv.Dataset(ds_wide.samples.T, sa=ds_wide.fa.copy(), fa=ds_wide.sa.copy())
    mv.h5save(results_dir + 'ses-localizer_task-objectcategories_ROIs_space-custom-subject_desc-highpass_transposed.hdf5', ds)
    print('Transposed the group-dataset and saved it.')
    return ds


def dotheclassification(ds, store_sens):
    """ Dotheclassification does the classification. It builds a
    linear gaussian naive bayes classifier, performs a leave-one-out
    crossvalidation and stores the sensitivities from the GNB classifier of each
    fold in a combined dataset for further use in a glm.
    If sens == False, the sensitivities are not stored, and only a
    classification is performed"""
    # set up classifier
    prior='ratio'
    targets= 'all_ROIs'
    gnb = mv.GNB(common_variance=True, prior=prior, space=targets)
    # prepare for callback of sensitivity extraction within CrossValidation
    sensitivities=[]
    if store_sens:
        def store_sens(data, node, result):
            sens = node.measure.get_sensitivity_analyzer(force_train=False)(data)
            # we also need to manually append the time attributes to the sens ds
            sens.fa['time_coords']=data.fa['time_coords']
            sens.fa['chunks']=data.fa['chunks']
            sensitivities.append(sens)

    # do a crossvalidation classification
        cv = mv.CrossValidation(gnb, mv.NFoldPartitioner(attr='participant'),
                                errorfx=mv.mean_match_accuracy,
                                enable_ca=['stats'],
                                callback=store_sens)
    else:
        cv = mv.CrossValidation(gnb, mv.NFoldPartitioner(attr='participant'),
                                errorfx=mv.mean_match_accuracy,
                                enable_ca=['stats'])
    results = cv(ds)
    # save classification results

    with open(results_dir + 'objectcategory_clf.txt', 'a') as f:
        f.write(cv.ca.stats.as_string(description=True))
    # print the confusion matrix
    desired_order = ['brain', 'VIS', 'left LOC', 'right LOC', 'left OFA', 'right OFA', 'left FFA', 'right FFA', 'left EBA', 'right EBA', 'left PPA', 'right PPA']
    labels = get_known_labels(desired_order, cv.ca.stats.labels)

    # plot the confusion matrix with pymvpas build-in plot function currently fails
    ##cv.ca.stats.plot(labels = labels, numbers = True, cmap = 'gist_hear_r')
    ##plt.savefig(results_dir + 'confusion_matrix.png')
    print('accuracy is {}'.format(cv.ca.stats.stats['mean(ACC)']))
    # plot confusion matrix with seaborne
    plot_confusion(cv,
                   labels,
                   fn = results_dir + 'confusion_GNB.pdf',
                   ACC = cv.ca.stats.stats['mean(ACC)'],
		   vmax=100)

    mv.h5save(results_dir + 'gnb_cv_classification_results.hdf5', results)
    print('Saved the crossvalidation results.')
    if store_sens:
        mv.h5save(results_dir + 'sensitivities_nfold.hdf5', sensitivities)
        print('Saved the sensitivities.')
    # results now has the overall accuracy. results.samples gives the
    # accuracy per participant.
    # sensitivities contains a dataset for each participant with the
    # sensitivities as samples and class-pairings as attributes
    return sensitivities, cv

    ## this is old: using the seaborne function to plot
##matrix = cv.ca.stats.matrix[reorder_full][:, reorder_full]
##labels = np.array(['EV' if label=='VIS' else label for label in
##                  cv.ca.stats.labels])[reorder_full]
##  plot_confusion(matrix,
##                  labels,
##                  fn=basedir + 'confusion_GNB_{}.pdf'.format(zscore_name),
##                  ACC=cv.ca.stats.stats['mean(ACC)'])
    #save the results dataset
#
# def classification_rois_only(ds, zscore_name):
#     ds_ROIs = ds[(ds.sa.all_ROIs!='brain') & (ds.sa.all_ROIs!='overlap'), :]
#     targets = 'all_ROIs'
#     prior = 'ratio'
#     sensitivities_rois = []
#     def store_sens(data, node, result):
#         sens = node.measure.get_sensitivity_analyzer(force_train = False)(data)
#         sens.fa['time_coords']=data.fa['time_coords']
#         sens.fa['chunks']=data.fa['chunks']
#         sensitivities_rois.append(sens)
#
#     clf_rois = mv.GNB(space=targets, prior=prior, common_variance=True)
#     cv_rois = mv.CrossValidation(clf_rois,
#                                 mv.NFoldPartitioner(attr='participant'),
#                                 errorfx=mv.mean_match_accuracy,
#                                 enable_ca=['stats'],
#                                 callback=store_sens)
#     results_rois = cv_rois(ds_ROIs)
#     with open(basedir + 'cv_only_rois_{}.txt',format(zscore_name), 'a') as f:
#         f.write(cv_rois.ca.stats.as_string(description=True))
#     # plot and save the confusion matrix
#     matrix = cv_rois.ca.stats.matrix[reorder_roi_only][:, reorder_roi_only]
#     labels = np.array(['EV' if label=='VIS' else label for label in
#                 cv_rois.ca.stats.labels])[reorder_roi_only]
#     plot_confusion(matrix,
#                    labels,
#                    fn=basedir+'confusion_GNB_only_rois_{}.pdf'.format(zscore_name),
#                    ACC = cv_rois.ca.stats.stats['mean(ACC)'])
#     print('Calculated and saved the classification results on data excluding the rest of the brain.')
#     return sensitivities_rois, cv_rois

# # rerun with coordinates
# def classification_with_coords(ds):
#     """ classification_with_coords performs the classification on a dataset
#     without fmri data and only voxel coordinates. If full = True, it will use
#     the data including the rest of the brain, if full = False, it will only the
#     ROIs."""
#     ds_coords = get_voxel_coords(ds, append=True, zscore=True)
#     if full:
#         ds_coords = ds_coords[(ds_coords.sa.all_ROIs!='brain') &
#                     (ds_coords.sa.all_ROIs!='overlap'), :]
#         name_ex = 'all'
#     else:
#         name_ex = 'only_rois'
#     targets = 'all_ROIs'
#     clf_with_coords = mv.GNB(space=targets, prior='ratio', common_variance=True)
#
#     sensitivities_with_coords=[]
#     def store_sens(data, node, result):
#         sens = node.measure.get_sensitivity_analyzer(force_train=False)(data)
#         # we also need to manually append the time attributes to the sens ds
#         sens.fa['time_coords']=data.fa['time_coords']
#         sens.fa['chunks']=data.fa['chunks']
#         sensitivities_with_coords.append(sens)
#
#     cv_with_coords = mv.CrossValidation(clf_with_coords,
#                             mv.NFoldPartitioner(attr='participant'),
#                             errorfx=mv.mean_match_accuracy,
#                             enable_ca=['stats'])
#
#     results_with_coords = cv_with_coords(ds_coords)
#     # save results
#     with open(basedir + 'cv_with_coords_{}.txt'.format(name_ex), 'a') as f:
#         f.write(cv_with_coords.ca.stats.as_string(description=True))
#     # plot and save the confusion matrix
#     if full:
#         reorder = reorder_full
#     else:
#         reorder = reorder_rois_only
#     matrix = cv_with_coords.ca.stats.matrix[reorder][:, reorder]
#     labels = np.array(['EV' if label=='VIS' else label for label in
#                 cv_with_coords.ca.stats.labels])[reorder]
#     plot_confusion(matrix,
#                    labels=labels,
#                    fn=basedir + 'confusion_GNB_with_coords_{}.pdf'.format(name_ex), \
#                    ACC = cv_with_coords.ca.stats.stats['mean(ACC)'])
#     print('Calculated and saved the classification results on data including \
#         coordinates.')


def dotheglm(sensitivities):
    """dotheglm does the glm. It will squish the sensitivity
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
    events = get_group_events()
    # save the event_file
    fmt = "%10.3f\t%10.3f\t%16s\t%60s"
    np.savetxt(results_dir + 'group_events.tsv', events, delimiter='\t', comments='', \
     header='onset\tduration\ttrial_type\tstim_file', fmt=fmt)
    # get events into dictionary
    events_dicts = []
    for i in range(0, len(events)):
        dic = {'onset':events[i][0], 'duration':events[i][1], 'condition':events[i][2]}
        events_dicts.append(dic)

    hrf_estimates = mv.fit_event_hrf_model(mean_sens_transposed,
                                            events_dicts,
                                            time_attr='time_coords',
                                            condition_attr='condition',
                                            design_kwargs=dict(drift_model='blank'),
                                            glmfit_kwargs=dict(model='ols'),
                                            return_model=True)
    mv.h5save(results_dir + 'sens_glm_objectcategories_results.hdf5', hrf_estimates)
    print('calculated glm, saving results.')
    return hrf_estimates


### the last thing to do is to plot this
def makeaplot(events,
              mean_sens_transposed,
              hrf_estimates,
              roi_pair,
              cv):
    """
    This produces a time series plot for the roi class comparison specified in
    roi_pair such as roi_pair = ['left FFA', 'left PPA']
    """
    import matplotlib.pyplot as plt
    # some parameters
    # get the conditions
    block_design = np.unique(events['trial_type'])
    # end indices to chunk timeseries into runs
    run_startidx= np.array([0, 157, 313, 469])
    run_endidx = np.array([156, 312, 468, 624])
    # more complex block design?
    runs = np.unique(mean_sens_transposed.sa.chunks)
    time_coords = mean_sens_transposed.sa.time_coords

    # plot sensitivity timecourse of particular comparison from one fold:
    for run in runs:
         # get a selection of matplotlib colors
        colors = ['#7b241c', '#e74c3c', '#154360', '#3498db', '#145a32','#27ae60',
        '#9a7d0a', '#f4d03f', '#5b2c6f', '#a569bd', '#616a6b', '#ccd1d1']
        fig, ax = plt.subplots(1, 1, figsize=[18, 10])
        plt.suptitle('Timecourse of sensitivities, {} versus {}, run {}'.format(roi_pair[0], roi_pair[1], run+1), fontsize='large')
        plt.xlim([0, 350])
        plt.xlabel('Time in sec')
        plt.legend()
        for stimulus in block_design:
            # get design information from the event file
            onsets = events[events['trial_type']==stimulus]['onset'] # stimulus start
            durations = events[events['trial_type']==stimulus]['duration']
            stimulation_end = np.sum([onsets, durations], axis=0) # stimulus end
            # I want the duration of the stimulation color filled
            for i in range(0, len(onsets)):
                ax.axvspan(onsets[i], stimulation_end[i], color=colors[0],
                alpha=0.5, label = "_"*i + stimulus)
            del colors[0]
            ax.legend()
        colors = ['#7b241c', '#e74c3c', '#154360', '#3498db', '#145a32','#27ae60',
        '#9a7d0a', '#f4d03f', '#5b2c6f', '#a569bd', '#616a6b', '#ccd1d1']
        for i in range(len(sensitivities[0])):
            comparison = (sensitivities[0][i].sa.items()[0][1].value[0])
            if (roi_pair[0] in comparison) and (roi_pair[1] in comparison):
                sens_targets = sensitivities[0][i].samples
                times = sensitivities[0][i].fa.time_coords[run_startidx[run]:run_endidx[run]]
                run_coords = np.array((times, sens_targets[0][run_startidx[run]:run_endidx[run]]))
                ax.plot(run_coords[0], run_coords[1], '-', color='black')
                # plot glm model results
                glm_model=hrf_estimates.a.model.results_[0.0].predicted[run_startidx[run]:run_endidx[run], i]
                ax2 = ax.twinx()
                ax2.plot(times, glm_model, '-', color = '#7b241c', lw=1)
                model_fit = hrf_estimates.a.model.results_[0.0].R2[i]
                del colors[0]
                acc=cv.ca.stats.stats['ACC%']
                plt.title('R squared: {}, accuracy: {}'.format(model_fit, acc))
                plt.savefig(basedir + '{}_vs_{}_run-{}'.format(roi_pair[0],
                            roi_pair[1], run))


# some sanity checks
def sanity_checking(ds):
    ds_onlycoords = get_voxel_coords(ds, append=False, zscore=False)
    sensititivities_only_coords = dotheclassification(ds_onlycoord,
                                                      zscore_name = 'sanity_check',
                                                      store_sens=False)
    classification_rois_only(ds_onlycoords, zscore_name = 'sanity_check')



if __name__ == '__main__':
    import os.path
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-z', '--z_score', help="How should the data be z-scored? \
                        'z-score' = normal z-scoring, 'custom' = zscoring based on \
                        activation during rest, 'no-zscoring' = Not at all",
                        type = str, default='z-score')
    parser.add_argument('-zc', '--custom_zscore', help='Should the data be \
                        z-scored with parameters derived from resting data? \
                        (0: True / 1: False)', type = int, default = 1)
    parser.add_argument('-ds', '--dataset', help="Specify whether the analysis \
                        should be done on the full dataset or on the dataset \
                        with only ROIs: 'full' or 'strippend'", type=str,
                        default = 'stripped')
    parser.add_argument('-c', '--coords', help="Should coordinates be included in \
                        the dataset? ('with-coordinates').Should a sanity check \
                        with only coordinates without fmri data be performed? \
                        ('only-coordinates'). Should coordinates be disregard? \
                        ('no-coordinates') Default: 'no-coordinates'.", type=str,
                        default = 'no-coordinates')
    parser.add_argument('-o', '--output', help="Please specify an output directory" 
                        "name (absolute path) to store the analysis results", type = str)

    args = parser.parse_args()
    # TODO: integrate this into functions
    zscore = args.z_score

    # this parameter should be used in the classification function to specify
    # whether the full dataset or a stripped dataset should be used.
    # TODO: integrate this into functions
    ds_type = args.dataset

    # this parameter should guide whether any analysis with coordinates should be performed
    coords = args.coords
    results_dir = '/' + args.output + '/'
    # the default is to get the sensitivities and compute a glm
    store_sens = True
    glm = True
    # build a dataset
    ds = buildadataset(zscore)
    # strip it if necessary
    if ds_type == 'stripped':
        ds = strip_ds(ds)
    # append coordinates if specified
    if coords == 'with-coordinates':
        ds = get_voxel_coords(ds, append=True, zscore=True)
        store_sens = False
	glm = False
        ## TODO: Do I at one point want to append the time_coordinates also to
        ## TODO: the dataset with coordinates?
    # of append coordinates and get rid of fmri data is specified
    elif coords == 'only-coordinates':
        ds = get_voxel_coords(ds, append=False, zscore=False)
        # if there is no fmri data in the ds, don't attempt to
        #  get sensitivities and only to a classification
        store_sens = False
        glm = False

    sensitivity, cv = dotheclassification(ds, store_sens=store_sens)

    if glm:
        hrf_estimates = dotheglm(sensitivity)


    ## TODO: This needs work now.
    # check whether data for subjects exists already, so that we can skip
    # # buildadataset()
    # groupdata = basedir + 'ses-localizer_task-objectcategories_ROIs_space-custom-subject_desc-highpass_transposed.hdf5'    sensdata = basedir + 'sensitivities_nfold.hdf5'
    # ev_file = basedir + 'group_events.tsv'
    # if os.path.isfile(groupdata):
    #     ds = mv.h5load(groupdata)
    # else:
    #     ds, zcsore_name = buildthisshit(zscore = zscore, zscore_custom = zscore_custom)
    # if (os.path.isfile(sensdata)) and (os.path.isfile(ev_file)):
    #     sensitivities = mv.h5load(sensdata)
    #     events = np.genfromtxt(ev_file, names=('onset', 'duration',
    #     'trial_type', 'stim_file'), dtype=['<f8', '<f8', '|S18', '|S60'],
    #     skip_header=1)
    # else:
    #    sensitivities, cv = dotheclassification(ds, zscore_name, store_sens = True)
    #    sensitivities, cv = dotheclassification(ds)
    #    classification_rois_only(ds, zscore_name)
    #    classification_with_coords(ds, full=False)
    #    hrf_estimates = dotheglm(sensitivities)
    #    only_coords = sanity_checking(ds)
