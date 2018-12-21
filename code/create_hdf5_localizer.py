#!/usr/bin/python

"""
This script is intended to create a (transposed) groupdataset
of the localizer data. It takes data from all participants
objectcategory runs, and the ROI data, and uses pymvpa to stack everything.
It uses the data that is already in group space,
and that has previously been preprocessed.
A non-transposed and a transposed dataset are saved.
Previously, this was done by a function within the outdated
hdf5_loc script called buildadataset().

###################
#
# TODO: this script will break - have changed sub-* dirs to
# contain ses-* dirs as well. Insert ses-movie where necessary
# at one point to fix!
#
###################
"""

import numpy as np
import mvpa2.suite as mv
from glob import glob


def get_group_events(event_path):
    """This function simply gets a group event file from the single event files
    per participant"""
    event_files = sorted(glob(event_path))
    if len(event_files) == 0:
        print('I could not find event files in the specified event_path {}.'
              .format(event_path))

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
                           dtype=[('onset', float), ('duration', float), ('trial_type', '|S16'), ('stim_file', '|S60')])
    for row, val in itertools.izip(events, meanvals):
        row['onset'] = val
    for filename in event_files:
        d = np.genfromtxt(filename,
                          delimiter='\t',
                          names=True,
                          dtype=[('onset', float), ('duration', float), ('trial_type', '|S16'), ('stim_file', '|S60')])
        for i in range(0, len(d)):
            import numpy.testing as npt
            npt.assert_almost_equal(events['onset'][i], d['onset'][i], decimal=0)
            npt.assert_almost_equal(events['duration'][i], d['duration'][i], decimal=0)
            assert events['trial_type'][i] == d['trial_type'][i]

    # account for more variance by coding the first occurrence of each category
    i = 1
    while i < len(events):
        if i == 1:
            events[i - 1]['trial_type'] = events[i - 1]['trial_type'] + '_first'
            i += 1
        if events[i - 1]['trial_type'] != events[i]['trial_type']:
            events[i]['trial_type'] = events[i]['trial_type'] + '_first'
            i += 2
        else:
            i += 1
    return events


def extract_baseline(events, localizer_ds):
    """ This function extracts the mean and standard deviation for z-scoring
    from those times of the experiments with no stimulation. It is meant to be
    used within building of the group dataset."""
    # get last stimulation before break
    import bisect

    rest_periods_start = []
    rest_periods_end = []
    for i in range(len(events) - 1):
        if i == 0:
            rest_periods_start.append(0.0)
            rest_periods_end.append(events[i]['onset'])
        else:
            dur = events[i + 1]['onset'] - events[i]['onset']
            if dur > 5.0:
                rest_periods_start.append(events[i]['onset'] + events[i]['duration'])
                rest_periods_end.append(events[i + 1]['onset'])
    # append the last stimulation end, and the end of the scan
    rest_periods_start.append(events[-1]['onset'] + events[-1]['duration'])
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


def buildadataset(zscore,
                  rois,
                  event_path=None):
    """buildataset() will build and save participant-specific hdf5 datasets
    with all rois from preprocessed objectcategories data, stack them for a
    group dataset and save them, and transpose the group dataset and save it.
    The parameter 'zscore' determines whether and what kind of z-scoring
    should be performed."""
    print('I am building a dataset with the following option: {}.'.format(zscore))

    # get the participants and rois
    participants = sorted([path.split('/')[-1] for path in glob(base_dir + 'sub-*')])
    localizer_dss = []

    for participant in participants:
        localizer_fns = sorted(glob(base_dir + participant + locdir + \
                                    '{}_task-objectcategories_run-*_space-custom-subject_desc-highpass_bold.nii.gz'.format(
                                        participant)))
        mask_fn = base_dir + participant + anat_dir + 'brain_mask.nii.gz'
        assert len(localizer_fns) == 4
        localizer_ds = mv.vstack([mv.fmri_dataset(localizer_fn, mask=mask_fn, chunks=run)
                                  for run, localizer_fn in enumerate(localizer_fns)])

        localizer_ds.fa['participant'] = [participant] * localizer_ds.shape[1]
        print('loaded localizer data for participant {}.'.format(participant))

        # zscore the data with means and standard deviations from no-stimulation
        # periods
        if zscore == 'custom':
            events = get_group_events(event_path)
            means, stds = extract_baseline(events, localizer_ds)
            # zscore stuff
            mv.zscore(localizer_ds, params=(means, stds), chunks_attr='chunks')
            print('finished custom zscoring for participant {}.'.format(participant))
        elif zscore == 'z-score':
            mv.zscore(localizer_ds, chunks_attr='chunks')
            print('finished zscoring for participant {}.'.format(participant))
        else:
            print('I did not zscore.')

        all_rois_mask = np.array([['brain'] * localizer_ds.shape[1]]).astype('S10')
        for roi in rois:
            # Get filenames for potential right and left ROI masks
            if roi == 'VIS':
                roi_fns = sorted(glob(base_dir + participant + anat_dir + \
                                      '{0}_*_mask.nii.gz'.format(roi)))
            else:
                left_roi_fns = sorted(glob(base_dir + participant + anat_dir + \
                                           'l{0}_*_mask.nii.gz'.format(roi)))
                right_roi_fns = sorted(glob(base_dir + participant + anat_dir + \
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
                roi_fns = sorted(glob(base_dir + participant + anat_dir + '/{0}_*_mask.nii.gz'.format(roi)))
                roi_mask = np.sum([mv.fmri_dataset(roi_fn, mask=mask_fn).samples for roi_fn in roi_fns],
                                  axis=0)
                np.place(all_rois_mask, (roi_mask > 0) & (all_rois_mask != 'brain'), 'overlap')
                all_rois_mask[(roi_mask > 0) & (all_rois_mask != 'overlap')] = roi
        # Flatten mask into list
        all_rois_flat = list(all_rois_mask.ravel())
        # Assign ROI mask to localizer data feature attributes
        localizer_ds.fa['all_ROIs'] = all_rois_flat

        if save_per_subject:
            mv.h5save(base_dir + participant + locdir + \
                  '{}_ses-localizer_task-objectcategories_ROIs_space-custom-subject_desc-highpass.hdf5'.format(
                      participant), localizer_ds)
            print('Saved dataset for {}.'.format(participant))
        # join all datasets
        localizer_dss.append(localizer_ds)

    # save full dataset
    mv.h5save(results_dir + 'ses-localizer_task-objectcategories_ROIs_space-custom-subject_desc-highpass.hdf5',
              localizer_dss)
    print('saved the collection of all subjects datasets.')
    # squish everything together
    ds_wide = mv.hstack(localizer_dss)

    # transpose the dataset, time points are now features
    ds = mv.Dataset(ds_wide.samples.T, sa=ds_wide.fa.copy(), fa=ds_wide.sa.copy())
    mv.h5save(
        results_dir + 'ses-localizer_task-objectcategories_ROIs_space-custom-subject_desc-highpass_transposed.hdf5', ds)
    print('Transposed the group-dataset and saved it.')
    return ds


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--base_dir', help="Please specify the root of your"
                                                 "dataset (e.g. /data/movieloc/backup_store/saccs/",
                        required=True)
    parser.add_argument('-l', '--loc_dir', help="Please specify the directory under "
                                                 "which the loc data can be found (e.g. /ses-localizer/func/",
                        required=True)
    parser.add_argument('-a', '--anat_dir', help="Please specify the directory under "
                                                 "which the anatomical data (ROIs) of "
                                                 "participants can be found (e.g. /ses-movie/anat/",
                        required=True)
    parser.add_argument('-r', '--results_dir', help="Please specify the directory under which the"
                                                    "resulting groupdatafiles should be saved.",
                        required=True)
    parser.add_argument('-z', '--zscoring', help="How should the data be z-scored? "
                                                 "'z-score' = normal z-scoring,"
                                                 "'custom' = zscoring based on activation during rest,"
                                                 "'no-zscoring' = Not at all",
                        type=str, default='z-score')
    parser.add_argument('-e', '--event_path', help="If custom zscoring is selected, the script needs to extract"
                                                   "data from rest periods. Therefore, please specify a path from"
                                                   "the root of the directory (base_dir) to"
                                                   "the subjects event file, to retrieve this information. Specify"
                                                   "a path with wildcards, such as"
                                                   " '/sourcedata/phase2/*/ses-localizer/func/sub-*_ses-localizer_task-objectcategories_run-*_events.tsv'")
    parser.add_argument('-s', '--save_individual_data', help="Please specify whether you want each participants"
                                                             "individual dataset to be saved (in the respective"
                                                             "participants ses-movie/func/ directory)(True, False)",
                         default=False)
    parser.add_argument('-R', '--ROIs', nargs='+', help="Supply all ROIs you want to include "
                                                        "(as --ROIs 'VIS' 'FFA' ...). If no ROIs are provided,"
                                                        "the dataset will include all ROIs.")

    args = parser.parse_args()

    base_dir = args.base_dir + '/'
    loc_dir = '/' + args.loc_dir + '/'
    anat_dir = '/' + args.anat_dir + '/'
    results_dir = '/' + args.results_dir + '/'
    save_per_subject = args.save_individual_data
    if args.ROIs:
        rois = [str(roi) for roi in args.ROIs]
    else:
        rois = ['FFA', 'OFA', 'PPA', 'EBA', 'LOC', 'VIS']
    zscore = args.zscoring
    if zscore == 'custom' and not args.event_path:
        print('Custom zscoring only works when this script is given event information.'
              'Please specify the --event_path flag with a path to the event files, such as'
              '/sourcedata/phase2/*/ses-localizer/func/sub-*_ses-localizer_task-objectcategories_run-*_events.tsv.'
              'For now, I will default to normal z-scoring.')
        zscore = 'z-score'
    if zscore == 'custom' and args.event_path:
        event_path = args.event_path
        buildadataset(zscore, rois=rois, event_path=event_path)

    else:
        buildadataset(zscore, rois, event_path=None)
