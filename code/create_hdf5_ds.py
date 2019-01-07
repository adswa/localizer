#!/usr/bin/env python

import numpy as np
import mvpa2.suite as mv
from glob import glob

"""
This is a general skript to create an hdf5 dataset for the avmovie or localizer data. The
inputs and results should in a group dataset space
It takes data from all participants, and their ROI data, stacks everything into an
hdf5 dataset, and transposes it.

Inputs are:
- --rootdir: The root of the directory, containing sub-* subdirectories. (defaults to '.')
- --datadir: The directory in which the preprocessed datafiles in group space can be found, 
             based on root (e.g. '/ses-localizer/func/sub-*_task-objectcategories_run-*_space-custom-subject_desc-highpass_bold_tmpl.nii.gz'
             or '/ses-movie/func/sub-01_ses-movie_task-avmovie_run-1_desc-highpass_tmpl.nii.gz)
- --anatdir: The directory where masks can be found, based on the individual subjects dirs,
             (e.g. 'ses-movie/anat/')
- --analysis: which dataset is being created: 'avmovie' or 'localizer'
- --z-scoring: Which type of zscoring to perform: 'no-zscore', 'zscore' or in the case of
                localizer data, 'baseline-zscore'
- --eventdir: If localizer data should be baseline zscored, specify a path to the directory
                of the experiments event files.
- --outdir: Where the results should be saved. Make sure its an absolute path!
- --rois
    parser.add_argument("-t", "--type", help="Which dataset is analysed? 'localizer' or 'avmovie'.",
                        required=True)
    parser.add_argument("-r", "--rois", help=" Specify a list of ROIs (e.g. 'FFA' 'PPA' 'EBA') to"
                                             "only include specific ROIs. If nothing is specified, all ROIs are used.",
"""


def extract_baseline(events,
                     data_ds):
    """
    Take man and standard deviation of baseline fmri data (between presentations of
    stimuli) from the localizer data to z-score with them.
    :param events:
    :param data_ds:
    :return:
    """
    import bisect

    rest_periods_start = []
    rest_periods_end = []
    for i in range(len(events) - 1):
        if i == 0:
            rest_periods_start.append(0.0)
            rest_periods_end.append(events[i]['onset'])
        else:
            dur = events[i + 1]['onset'] - events[i]['onset']
            # stimulation periods are followed by short breaks - I want these
            if dur > 5.0:
                rest_periods_start.append(events[i]['onset'] + events[i]['duration'])
                rest_periods_end.append(events[i + 1]['onset'])
    # append the last stimulations end, and the end of the scan
    rest_periods_start.append(events[-1]['onset'] + events[-1]['duration'])
    rest_periods_end.append(data_ds.sa.time_coords[-1])
    assert len(rest_periods_start) == len(rest_periods_end)

    # extract the activation within the time slots and compute mean and std.
    # a bit tricky as event file and activation data exist in different time
    # resolutions. Will settle for an approximate solution, where I extract the
    # time coordinate closest to the onset and offset of restperiods
    restdata = []
    for i in range(len(rest_periods_start)):
        start_idx = bisect.bisect_left(data_ds[data_ds.sa.chunks ==
                                               0].sa.time_coords, rest_periods_start[i])
        end_idx = bisect.bisect_left(data_ds[data_ds.sa.chunks ==
                                             0].sa.time_coords, rest_periods_end[i])
        restdata.append(data_ds.samples[start_idx:end_idx])
    # flatten the list of arrays
    rests = np.concatenate(restdata)
    # get mean and std as per-feature vectors
    means = np.mean(rests, axis=0)
    std = np.std(rests, axis=0)
    return means, std


def get_group_events(eventdir):
    """
    If we analyze the localizer data, this function is necessary
    to average all event files into one common event file.
    """
    import itertools

    event_files = sorted(glob(eventdir + '*_events.tsv'))
    if len(event_files) == 0:
        print("I could not find event files in the event directory")
    assert len(event_files) > 0

    # compute the average of the event files to get a general event file
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
                           dtype=[('onset', float),
                                  ('duration', float),
                                  ('trial_type', '|S16'),
                                  ('stim_file', '|S60')])
    for row, val in itertools.izip(events, meanvals):
        row['onset'] = val
    for filename in event_files:
        d = np.genfromtxt(filename,
                          delimiter='\t',
                          names=True,
                          dtype=[('onset', float),
                                 ('duration', float),
                                 ('trial_type', '|S16'),
                                 ('stim_file', '|S60')])
        for i in range(0, len(d)):
            # assert that no individual stimulation protocol deviated from the
            # average by more than a second, assert that trial ordering did not
            # get confused
            import numpy.testing as npt
            npt.assert_almost_equal(events['onset'][i], d['onset'][i], decimal=0)
            npt.assert_almost_equal(events['duration'][i], d['duration'][i], decimal=0)
            assert events['trial_type'][i] == d['trial_type'][i]

    # account for more variance by coding the first occurrence in each category in a new event
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

    # returns an event file array
    return events


def createdataset(analysis,
                  datadir,
                  rootdir,
                  anatdir,
                  eventdir,
                  zscore,
                  rois):
    """
    Build an hdf5 dataset.
    """
    # initialize a list to load all datasets into:
    data_dss = []

    # get list of participants from root dir
    participants = sorted([path.split('/')[-1] for path in glob(rootdir + 'sub-*')])
    assert len(participants) != 0
    print('The following participants were found: {}'.format(participants))

    for participant in participants:
        # count the number of participant substitutions necessary
        data_fns = sorted(glob(rootdir + participant + datadir))
        print(rootdir + participant + datadir)
        mask_fn = rootdir + participant + anatdir + 'brain_mask_tmpl.nii.gz'
        if analysis == 'localizer':
            assert len(data_fns) == 4
        if analysis == 'avmovie':
            assert len(data_fns) == 8
        data_ds = mv.vstack([mv.fmri_dataset(data_fn, mask=mask_fn, chunks=run)
                             for run, data_fn in enumerate(data_fns)])
        data_ds.fa['participant'] = [participant] * data_ds.shape[1]
        print('loaded data for participant {}.'.format(participant))

        # z scoring
        if analysis == 'localizer' and zscore == 'baseline-zscore':
            events = get_group_events(eventdir)
            means, stds = extract_baseline(events, data_ds)
            mv.zscore(data_ds, params=(means, stds), chunks_attr='chunks')
            print('finished baseline zscoring for participant {}.'.format(participant))
        elif zscore == 'zscore':
            mv.zscore(data_ds, chunks_attr='chunks')
            print('finished zscoring for participant {}.'.format(participant))
        else:
            print('I did not zscore.')

        # roi masks
        all_rois_mask = np.array([['brain'] * data_ds.shape[1]]).astype('S10')
        for roi in rois:
            # Get filenames for potential right and left ROI masks
            if roi == 'VIS':
                roi_fns = sorted(glob(rootdir + participant + anatdir + \
                                      '{0}_*_mask_tmpl.nii.gz'.format(roi)))
            else:
                left_roi_fns = sorted(glob(rootdir + participant + anatdir + \
                                           'l{0}_*_mask_tmpl.nii.gz'.format(roi)))
                right_roi_fns = sorted(glob(rootdir + participant + anatdir + \
                                            'r{0}_*_mask_tmpl.nii.gz'.format(roi)))
                roi_fns = left_roi_fns + right_roi_fns
            if len(roi_fns) == 0:
                print("ROI {0} does not exist for participant {1}; appending all zeros".format(roi, participant))
                roi_mask = np.zeros((1, data_ds.shape[1]))
            elif len(roi_fns) == 1:
                roi_mask = mv.fmri_dataset(roi_fns[0], mask=mask_fn).samples
            elif len(roi_fns) > 1:
                # Add ROI maps into single map
                print("Combining {0} {1} masks for participant {2}".format(
                    len(roi_fns), roi, participant))
                roi_mask = np.sum([mv.fmri_dataset(roi_fn, mask=mask_fn).samples for roi_fn in roi_fns], axis=0)
                # Set any voxels that might exceed 1 to 1
                roi_mask = np.where(roi_mask > 0, 1, 0)

            # Ensure that number of voxels in ROI mask matches dataset dimension
            assert roi_mask.shape[1] == data_ds.shape[1]
            # Flatten mask into list
            roi_flat = list(roi_mask.ravel())
            # Assign ROI mask to data feature attributes
            data_ds.fa[roi] = roi_flat
            # Get lateralized masks as well
            if roi != 'VIS':
                lat_roi_mask = np.zeros((1, data_ds.shape[1]))
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
                    left_roi_mask = np.zeros((1, data_ds.shape[1]))

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
                    right_roi_mask = np.zeros((1, data_ds.shape[1]))

                # Ensure that number of voxels in ROI mask matches dataset dimension
                assert lat_roi_mask.shape[1] == data_ds.shape[1]
                # Flatten mask into list
                lat_roi_flat = list(lat_roi_mask.ravel())
                # Assign ROI mask to data feature attributes
                data_ds.fa['lat_' + roi] = lat_roi_flat
                # Check existing feature attribute for all ROIS for overlaps
                np.place(all_rois_mask, ((left_roi_mask > 0) | (right_roi_mask > 0))
                         & (all_rois_mask != 'brain'), 'overlap')

                all_rois_mask[(left_roi_mask > 0) & (all_rois_mask != 'overlap')] = 'left {0}'.format(roi)
                all_rois_mask[(right_roi_mask > 0) & (all_rois_mask != 'overlap')] = 'right {0}'.format(roi)
            elif roi == 'VIS':
                roi_fns = sorted(glob(rootdir + participant + anatdir + '/{0}_*_mask_tmpl.nii.gz'.format(roi)))
                roi_mask = np.sum([mv.fmri_dataset(roi_fn, mask=mask_fn).samples for roi_fn in roi_fns], axis=0)
                np.place(all_rois_mask, (roi_mask > 0) & (all_rois_mask != 'brain'), 'overlap')
                all_rois_mask[(roi_mask > 0) & (all_rois_mask != 'overlap')] = roi

        # Flatten mask into list
        all_rois_flat = list(all_rois_mask.ravel())

        # Assign roi mask to dataset feature attributes
        data_ds.fa['all_ROIs'] = all_rois_flat

        # join all datasets
        data_dss.append(data_ds)

    # save full dataset
    mv.h5save(outdir + '{}_groupdataset.hdf5'.format(analysis), data_dss)
    print('saved the collection of all subjects datasets.')
    # squish everything together
    ds_wide = mv.hstack(data_dss)
    # transpose the dataset, time points are now features
    ds = mv.Dataset(ds_wide.samples.T, sa=ds_wide.fa.copy(), fa=ds_wide.sa.copy())
    mv.h5save(outdir + '{}_groupdataset_transposed.hdf5'.format(analysis), ds)
    print('Transposed the group-dataset and saved it.')
    return ds


if __name__ == '__main__':
    import argparse
    import os

    parser = argparse.ArgumentParser()

    parser.add_argument('-r', '--rootdir', help="What is the root of the dataset?"
                                                "(Where sub-* directories are located?). Defaults to '.'.",
                        default='.')
    parser.add_argument('-d', '--datadir', help="Where are the bold.nii.gz files"
                                                "located, based of the subjects subdirectory? (e.g. ", required=True)
    # ideally, all data should live inside each subjects sub-directory. for now (bigfixxing)
    # I'll go back to sourcedata. 'sourcedata/aligned/{}/in_bold3Tp2/{}_task-objectcategories_run-*_bold.nii.gz')
    parser.add_argument('-a', '--anatdir', help="Where are the necessary masks"
                                                "located, based of the subjects subdirectory?"
                                                " (e.g. '/ses-movie/anat/')",
                        required=True)
    parser.add_argument('-e', '--eventdir', help="I the localizer dataset is used with"
                                                 "baseline zscoring, specify a path to the directory with the event"
                                                 "files based on root directory or as an absolute path. "
                                                 "(e.g. 'sourcedata/phase2/*/ses-localizer/func/")
    parser.add_argument('-z', '--zscoring', help="What kind of z-scoring should be done?"
                                                "'zscore', 'no-zscore', 'baseline-zscore' (the latter option is"
                                                "only available for the localizer data and needs directory of"
                                                "the stimulations eventfiles in --eventdir).", default='zscore')
    parser.add_argument("-t", "--type", help="Which dataset is analysed? 'localizer' or 'avmovie'.",
                        required=True)
    parser.add_argument("--rois", help=" Specify a list of ROIs (e.g. 'FFA' 'PPA' 'EBA') to"
                                             "only include specific ROIs. If nothing is specified, all ROIs are used.",
                        nargs='+')
    parser.add_argument('-o', '--output', help="Where should the resulting datafile be"
                                               "saved? Defaults to root directory. If you specify something,"
                                               "make sure its an absolute path!")

    args = parser.parse_args()

    rootdir = args.rootdir + '/'
    anatdir = '/' + args.anatdir + '/'
    datadir = '/' + args.datadir
    zscore = args.zscoring
    if zscore == 'baseline-zscore' and not args.eventdir:
        print('Custom zscoring only works when this script is given event information.'
              'Please specify the --event_path flag with a path to the event files, such as'
              '/sourcedata/phase2/*/ses-localizer/func/'
              'For now, I will default to normal z-scoring.')
        eventdir = None
        zscore = 'zscore'
    elif zscore == 'baseline-zscore' and args.eventdir:
        eventdir = args.eventdir + '/'
    else:
        eventdir = None

    if args.output:
        outdir = args.output + '/'
    else:
        outdir = rootdir
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    allowed_types = ['localizer', 'avmovie']
    analysis = args.type
    if analysis not in allowed_types:
        print("You specified analysis type {0}, "
              "however, the possible analysis types are {1}".format(analysis, allowed_types))
    assert analysis in allowed_types

    if args.rois:
        rois = [i for i in args.rois]
    else:
        rois = ['FFA', 'OFA', 'PPA', 'EBA', 'LOC', 'VIS']

    createdataset(analysis=analysis,
                  datadir=datadir,
                  rootdir=rootdir,
                  anatdir=anatdir,
                  eventdir=eventdir,
                  zscore=zscore,
                  rois=rois)

