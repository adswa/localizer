#!/usr/bin/python


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
def runthisshit(roi_pairs):

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

    # squish everything together
    ds_wide = mv.hstack(movie_dss)

    # transpose the dataset, time points are now features
    ds = mv.Dataset(ds_wide.samples.T, sa=ds_wide.fa.copy(), fa=ds_wide.sa.copy())
    mv.h5save(basedir + 'ses-localizer_task-objectcategories_ROIs_space-custom-subject_desc-highpass_transposed.hdf5', ds)


    #########
    # step 2:
    # we have a dataset, lets start the classification
    prior='ratio'
    targets= 'all_ROIs'
    gnb = mv.GNB(common_variance=True, prior=prior, space=targets)

    # prepare for sensitivity extraction within CrossValidation
    # how can I be sure which participant was used?
    sensitivities=[]
    def store_sens(data, node, result):
        sens = node.measure.get_sensitivity_analyzer(force_train=False)(data)
        # we also need to manually append the time attributes to the sens ds
        sens.fa['time_coords']=data.fa['time_coords']
        # and maybe run information? partipant?
        sens.fa['chunks']=data.fa['chunks']
        sensitivities.append(sens)

    # do a crossvalidation classification
    cv = mv.CrossValidation(gnb, mv.NFoldPartitioner(attr='participant'),
                            errorfx=mv.mean_match_accuracy,
                            enable_ca=['stats'],
                            callback=store_sens)
    results = cv(ds)
    # save classification results
    with open('/data/movieloc/backup_store/gnb_cv_results/objectcategory_clf.txt', 'a') as f:
        f.write(cv.ca.stats.as_string(description=True))
    #save the results dataset
    mv.h5save(basedir + '/gnb_cv_results/' + 'gnb_classification_results', results)
    # results now has the overall accuracy. results.samples gives the
    # accuracy per participant.
    # sensitivities contains a dataset for each participant with the
    # sensitivities as samples and class-pairings as attributes

    ########
    # step 3:
    # get event files. they're located in sourcedata/phase2/sub-*/ses-localizer/func/sub*_ses-localizer_task-objectcategories_run*_events.tsv
    # TODO: move event files into subs ses-localizer dir

    ## CODE TO CALCULATE ONE GLM PER SUBJECT?
    for sub in range(0, len(participants)):
        participant = participants[sub]
        event_files = sorted(glob(basedir + '/sourcedata/phase2/{}/ses-localizer/func/sub-*_ses-localizer_task-objectcategories_run-*_events.tsv'.format(participant)))
        assert len(event_files)==4
        #make them more detailed
        print('starting analysis of {}.'.format(participant))
        for event_file in sorted(event_files):
            data = np.genfromtxt(event_file, dtype=[('onset',float), ('duration', float),('trial_type', '|S16'), ('stim_file', '|S60')], 
                                 delimiter='\t', names=True)
            # simplify data
            i = 1
            while i < len(data):
                if i == 1:
                    data[i-1]['trial_type'] = data[i-1]['trial_type'] + '_first'
                    i += 1
                if data[i-1]['trial_type'] != data[i]['trial_type']:
                    data[i]['trial_type'] = data[i]['trial_type'] + '_first'
                    i += 2
                else:
                    i += 1
            # get events into dictionary
            dicts = []
            for i in range(0, len(data)):
                dic = {'onset':data[i][0], 'duration':data[i][1], 'condition':data[i][2]}
                dicts.append(dic)
            #this should contain events for all four runs
        ds = sensitivities[sub]
        import itertools
        for pair in itertools.combinations(roi_pairs, 2):
                                        ##    ['left FFA', 'left PPA', 'right FFA',
                                        ##    'right PPA'], 2):
                                        #    'left EBA', 'left FFA', 'left LOC',
                                        #    'left OFA', 'left PPA', 'right EBA',
                                        #    'right FFA', 'right LOC', 'right OFA',
                                        #    'right PPA', 'VIS'], 2):
            t1=pair[0]
            t2=pair[1]
            for i in range(0, len(ds.sa.all_ROIs)):
                if (t1 in ds.sa.all_ROIs[i]) and (t2 in ds.sa.all_ROIs[i]):
                    row_idx = i
            y = ds[row_idx] # check whether this selection works
            t_1 = t1.split(' ')
            t_2 = t2.split(' ')
            names = t_1 + t_2
            model_name='-'.join(names)
            # transpose dataset because fit_event_hrf_model expects time_coords as
            # sample attributes
            y_T = mv.Dataset(y.samples.T, sa = y.fa.copy(), fa=y.sa.copy())
            hrf_estimates = mv.fit_event_hrf_model(y_T,
                                                   dicts,
                                                   time_attr='time_coords',
                                                   condition_attr='condition',
                                                   design_kwargs=dict(drift_model='blank'),
                                                   glmfit_kwargs=dict(model='ols'),
                                                   return_model=True)

            print('calculated glm for {}, saving results...'.format(participant))
            #save results under model_name at some point
            mv.h5save('/data/movieloc/backup_store/gnb_cv_results/{}-{}'.format(participant, model_name), hrf_estimates)
    print('I am done')



    # ## CODE TO CALCULATE ONE GLM PER RUN
    #
    # for sub in range(0, len(participants)):
    #     participant = participants[sub]
    #     event_files = sorted(glob(basedir +
    #                         '/sourcedata/phase2/{}/ses-localizer/func/sub-*_ses-localizer_task-objectcategories_run-*_events.tsv'.format(participant)))
    #     assert len(event_files)==4
    #     #make them more detailed
    #     print('starting analysis of {}.'.format(participant))
    #     for event_file in sorted(event_files):
    #         data = np.genfromtxt(event_file, dtype=[('onset',float), ('duration', float),('trial_type', '|S16'), ('stim_file', '|S60')], 
    #                              delimiter='\t', names=True)
    #         # simplify data
    #         i = 1
    #         while i < len(data):
    #             if i == 1:
    #                 data[i-1]['trial_type'] = data[i-1]['trial_type'] + '_first'
    #                 i += 1
    #             if data[i-1]['trial_type'] != data[i]['trial_type']:
    #                 data[i]['trial_type'] = data[i]['trial_type'] + '_first'
    #                 i += 2
    #             else:
    #                 i += 1
    #         # get events into dictionary
    #         dicts = []
    #         for i in range(0, len(data)):
    #             dic = {'onset':data[i][0], 'duration':data[i][1], 'condition':data[i][2]}
    #             dicts.append(dic)
    #
    #         #run a glm. events files are now per run.
    #         # I need to get the sensitivities for a specific class decision (i.e. right
    #         # FFA versus right PPA) (thats one row in the ds.samples) for the time
    #         # points of one particular run.
    #
    #         # get the dataset for the participant
    #         ds = sensitivities[sub]
    #         # get index of the relevant targets
    #
    #         # to get indices per run
    # #        for run in np.unique(ds.fa.chunks)
    # #           run_idx = np.where(ds.fa.chunks == run)[0]
    #         # get all pairwise roi decisions
    #         import itertools
    #         for pair in itertools.combinations(['left EBA', 'left FFA', 'left LOC',
    #                                             'left OFA', 'left PPA', 'right EBA',
    #                                             'right FFA', 'right LOC', 'right OFA',
    #                                             'right PPA', 'VIS'], 2):
    #             t1=pair[0]
    #             t2=pair[1]
    #             for i in range(0, len(ds.sa.all_ROIs)):
    #                 if (t1 in ds.sa.all_ROIs[i]) and (t2 in ds.sa.all_ROIs[i]):
    #                     row_idx = i
    #                 for j in range(0, 4):
    #                     col_idx = np.where(ds.fa.chunks==j)[0]
    #                     y = ds[row_idx, col_idx]
    #                     t_1 = t1.split(' ')
    #                     t_2 = t2.split(' ')
    #                     names = t_1 + t_2
    #                     names.append('_run')
    #                     names.append(str(j+1))
    #                     model_name='-'.join(names)
    #                     # problem: fit_event_hrf_model expects time coordinates to
    #                     # be a sample attribute, here however they are a feature
    #                     # attribute. maybe retranspose the dataset?
    #                     y_T = mv.Dataset(y.samples.T, sa = y.fa.copy(),
    #                                         fa=y.sa.copy())
    #                     hrf_estimates = mv.fit_event_hrf_model(y_T,
    #                                                     dicts,
    #                                                     time_attr='time_coords',
    #                                                     condition_attr='condition',
    #                                                     design_kwargs=dict(drift_model='blank'),
    #                                                     glmfit_kwargs=dict(model='ols'),
    #                                                     return_model=True)
    #
    #                     print('calculated glm for {}, saving results...'.format(participant))
    #                     #save results under model_name at some point
    #                     mv.h5save('/data/movieloc/backup_store/gnb_cv_results/{}-{}'.format(participant, model_name), hrf_estimates)
    #




if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--roi', nargs='+', #action = 'append',
                        help = 'which rois to make comparisons \
                        between. Build a list with all of the rois you want \
                        included. Chose from left FFA, right FFA, left EBA, \
                        right EBA, left LOC, right LOC, left OFA, right OFA, \
                        left PPA, right PPA, VIS. Specify as strings in a list')

    args = parser.parse_args()

#    if (type(args.roi)==list) and (len(args.roi) > 1):
#        roi_pairs = args.roi
#    else:
#        print('rois should be specified as a list of minimum length 2 \
#               however, I was given {} as input for --roi. Will be \
#               using all possible pairs of rois instead'.format(args.roi))
#        roi_pairs = ['left FFA', 'left EBA', 'left LOC', 'left OFA',
#                    'left PPA', 'right EBA', 'right FFA', 'right LOC', 'right OFA',
#                    'right PPA', 'VIS']
    #else:
    #    roi_pairs = ['left FFA', 'left EBA', 'left LOC', 'left OFA',
    #                 'left PPA', 'right EBA', 'right FFA', 'right LOC', 'right OFA',
    #                 'right PPA', 'VIS']
    roi_pairs = args.roi

    print(roi_pairs)
    runthisshit(roi_pairs)


