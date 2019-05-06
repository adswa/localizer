#!/usr/bin/env python
import os
import mvpa2.suite as mv
import numpy as np
import pandas as pd
from glob import glob

"""
The main script was getting too crowded to efficiently work on it.
I'm outsourcing the majority of the helper functions to this utils
collection.
"""


def strip_ds(ds, order='full'):
    """this helper takes a dataset with brain and overlap ROIs and strips these
    categories from it.
    order: specifies amount of stripping ('full' --> strips brain and overlap
                                          'sparse' --> strips only overlap)
    """
    if order == 'full':
        print("attempting to exclude any overlaps and rest of the brain from"
              "the dataset.")
        if 'brain' in np.unique(ds.sa.all_ROIs):
            ds = ds[(ds.sa.all_ROIs != 'brain'), :]
            assert 'brain' not in ds.sa.all_ROIs
            print('excluded the rest of the brain from the dataset.')
        if 'overlap' in np.unique(ds.sa.all_ROIs):
            ds = ds[(ds.sa.all_ROIs != 'overlap'), :]
            assert 'overlap' not in ds.sa.all_ROIs
            print('excluded overlap from the dataset.')
    if order == 'sparse':
        print("attempting to exclude any overlaps from the dataset.")
        if 'overlap' in np.unique(ds.sa.all_ROIs):
            ds = ds[(ds.sa.all_ROIs != 'overlap'), :]
            assert 'overlap' not in ds.sa.all_ROIs
            print('excluded overlap from the dataset.')
    return ds


def bilateralize(ds):
    """combine lateralized ROIs in a dataset."""
    ds_ROIs = ds.copy('deep')
    ds_ROIs.sa['bilat_ROIs'] = [label.split(' ')[-1] for label in ds_ROIs.sa.all_ROIs]
    print('Combined lateralized ROIs for the provided dataset.')
    return ds_ROIs


def get_known_labels(desired_order, known_labels):
    """ Helper function to reorder ROI labels in a confusion matrix."""
    return [
        label
        for label in desired_order
        if label in known_labels
    ]


def plot_confusion(cv,
                   labels,
                   fn=None,
                   figsize=(9, 9),
                   vmax=None,
                   cmap='gist_heat_r',
                   ACC=None,
                   TPR=None,
                   PPV=None):
    """ This function plots the classification results as a confusion matrix.
    Specify ACC as cv.ca.stats.stats['mean(ACC)']/TPR as cv.ca.stats.stats['TPR']/
    PPV as cv.ca.stats.stats['PPV'] to display accuracy in the
    title. Set a new upper boundery of the scale with vmax. To save the plot,
    specify a path/with/filename.png as the fn parameter. """

    import seaborn as sns
    import matplotlib.pyplot as plt
    origlabels = cv.ca.stats.labels
    origlabels_indexes = dict([(x, i) for i, x in enumerate(origlabels)])
    reorder = [origlabels_indexes.get(labels[i]) for i in range(len(labels))]
    matrix = cv.ca.stats.matrix[reorder][:, reorder].T
    # Plot matrix with color scaled to 90th percentile
    fig, ax = plt.subplots(figsize=figsize)
    im = sns.heatmap(100 * matrix.astype(float) / np.sum(matrix, axis=1)[:, None],
                     cmap=cmap,
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
        plt.suptitle('Mean accuracy: {}, Recall: {}, Precision: {}'.format(ACC, TPR, PPV))
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


def get_voxel_coords(ds,
                     append=True,
                     zscore=True):
    """ This function is able to append coordinates (and their
    squares, etc., to a dataset. If append = False, it returns
    a dataset with only coordinates, and no fmri data. Such a
    dataset is useful for a sanity check of the classification.
    """
    ds_coords = ds.copy('deep')
    # Append voxel coordinates (and squares, cubes)
    products = np.column_stack((ds.sa.voxel_indices[:, 0] * ds.sa.voxel_indices[:, 1],
                                ds.sa.voxel_indices[:, 0] * ds.sa.voxel_indices[:, 2],
                                ds.sa.voxel_indices[:, 1] * ds.sa.voxel_indices[:, 2],
                                ds.sa.voxel_indices[:, 0] * ds.sa.voxel_indices[:, 1] * ds.sa.voxel_indices[:, 2]))
    coords = np.hstack((ds.sa.voxel_indices,
                        ds.sa.voxel_indices ** 2,
                        ds.sa.voxel_indices ** 3,
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


def get_group_events(eventdir):
    """
    If we analyze the localizer data, this function is necessary
    to average all event files into one common event file.
    """
    import itertools

    event_files = sorted(glob(eventdir + '*_events.tsv'))
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

def norm_and_mean(norm,
                  bilateral,
                  classifier,
                  sensitivities):
    """This function normalizes a list of sensitivities to their
    L2 norm if norm = True, else just stacks them according to the
    classifier they were build with. Resulting stack of sensitivities
    is averaged with the mean_group_sample() function."""
    if norm:
        from sklearn.preprocessing import normalize
        import copy
        # default for normalization is the L2 norm
        sensitivities_to_normalize = copy.deepcopy(sensitivities)
        for i in range(len(sensitivities)):
            sensitivities_to_normalize[i].samples = normalize(sensitivities_to_normalize[i].samples, axis=1)

        sensitivities_stacked = mv.vstack(sensitivities_to_normalize)
        print('I normalized the data.')

    else:
        sensitivities_stacked = mv.vstack(sensitivities)

    sgds = ['sgd', 'l-sgd']

    if bilateral:
        if classifier in sgds:
            # Note: All SGD based classifier wanted an explicit
            # 'target' sample attribute, therefore, this is still present
            # in the sensitivities.
            #import pdb; pdb.set_trace()
            sensitivities_stacked.sa['bilat_ROIs_str'] = map(lambda p: '_'.join(p),
                                                             sensitivities_stacked.sa.targets)
        else:
            # ...whereas in GNB, the results are in 'bilat_ROIs' sample attribute
            sensitivities_stacked.sa['bilat_ROIs_str'] = map(lambda p: '_'.join(p),
                                                             sensitivities_stacked.sa.bilat_ROIs)
        mean_sens = mv.mean_group_sample(['bilat_ROIs_str'])(sensitivities_stacked)

    else:
        if classifier in sgds:
            # Note: All SGD based classifier wanted an explicit
            # 'target' sample attribute, therefore, this is still present
            # in the sensitivities.
            sensitivities_stacked.sa['all_ROIs_str'] = map(lambda p: '_'.join(p),
                                                           sensitivities_stacked.sa.targets)
        else:
            # ...whereas in GNB, the results are in 'all_ROIs' sample attribute
            sensitivities_stacked.sa['all_ROIs_str'] = map(lambda p: '_'.join(p),
                                                           sensitivities_stacked.sa.all_ROIs)
        mean_sens = mv.mean_group_sample(['all_ROIs_str'])(sensitivities_stacked)

    # return the averaged sensitivities
    return mean_sens


def get_roi_pair_idx(bilateral,
                     classifier,
                     roi_pair,
                     hrf_estimates,
                     ):
    """This is a helper function that retrieves the correct index for a specific roi
    pair decision from hrf_estimates based on the underlying dataset size and the
    used classifier."""
    sgds = ['sgd', 'l-sgd']

    if bilateral:
        for j in range(len(hrf_estimates.fa.bilat_ROIs_str)):
            if classifier in sgds:
                comparison = hrf_estimates.fa.targets[j][0]
            else:
                comparison = hrf_estimates.fa.bilat_ROIs[j][0]
            if (roi_pair[0] in comparison) and (roi_pair[1] in comparison):
                roi_pair_idx = j
    else:
        for j in range(len(hrf_estimates.fa.all_ROIs_str)):
            if classifier in sgds:
                comparison = hrf_estimates.fa.targets[j][0]
            else:
                comparison = hrf_estimates.fa.all_ROIs[j][0]
            if (roi_pair[0] in comparison) and (roi_pair[1] in comparison):
                roi_pair_idx = j
    return roi_pair_idx


def get_avmovietimes(mean_sens_transposed):
    """
    helper function to get TR, runonsets, and append proper
    time_coordinates to the sensitivities based on it (necessary for the
    avmovie analysis).

    Parameters:
        mean_sens_transposed: averaged transposed sensitivities
    """
    # TR was not preserved/carried through in .a
    # so we will guestimate it based on the values of time_coords
    tc = mean_sens_transposed.sa.time_coords
    TRdirty = sorted(np.unique(tc[1:] - tc[:-1]))[-1]
    assert np.abs(np.round(TRdirty, decimals=2) - TRdirty) < 0.0001

    # make time coordinates real seconds
    mean_sens_transposed.sa.time_coords = np.arange(len(mean_sens_transposed)) * TRdirty

    # get runs, and runlengths in seconds
    runs = sorted(mean_sens_transposed.UC)
    assert runs == range(len(runs))
    runlengths = [np.max(tc[mean_sens_transposed.sa.chunks == run]) + TRdirty
                  for run in runs]
    runonsets = [sum(runlengths[:run]) for run in runs]
    assert len(runs) == 8
    # check whether chunks are increasing as well as sanity check
    chunks = mean_sens_transposed.sa.chunks
    assert np.all(chunks[1:] >= chunks[:-1])

    return mean_sens_transposed, chunks, runs, runonsets


def get_events(analysis,
               eventdir,
               results_dir,
               chunks=False,
               runs=False,
               runonsets=False,
               annot_dir=False,
               multimatch=False):
    """
    Function to extract events from the provided annotations.
    Parameters:
        analysis: localizer or avmovie
        eventdir: path leading to the appropriate event file given the analysis type
        annot_dir: path leading to the location annotation, if available
        chunks, runs, runonsets = necessary for avmovie analysis

    """
    if analysis == 'avmovie':
        # We're building an event file from the location annotation and the face events

        if multimatch:
            # glob and sort the multimatch results
            multimatch_files = sorted(glob(multimatch))
            assert len(multimatch_files) == len(runs)
            multimatch_dfs = []
            # read in the files, and make sure we get the onsets to increase. So
            # far, means.csv files always restart onset as zero.
            # the onsets restart every new run from zero, we have to append the
            # runonset times:
            for idx, multimatch_file in enumerate(multimatch_files):
                data = pd.read_csv(multimatch_file, sep='\t')
                data['onset'] += runonsets[idx]
                multimatch_dfs.append(data)

            # get everything into one large df
            mm = pd.concat(multimatch_dfs).reset_index()
            assert np.all(mm.onset[1:].values >= mm.onset[:-1].values)
            # get the duration and position similarity measures from multimatch
            # zcore the Position and Duration results around mean 1. We use
            # those because of a suboptimal correlation structure between the
            # similarity measures.
            from scipy import stats
            dur_sim = stats.zscore(mm.duration_sim) + 1
            pos_sim = stats.zscore(mm.position_sim) + 1
            onset = mm.onset.values

            # put them into event file structure
            dur_sim_ev = pd.DataFrame({
                'onset': onset,
                'duration': mm.duration.values,
                'condition': ['duration_sim'] * len(mm),
                'amplitude': dur_sim
            })

            pos_sim_ev = pd.DataFrame({
                'onset': onset,
                'duration': mm.duration.values,
                'condition': ['position_sim'] * len(mm),
                'amplitude': pos_sim
            })
            # sort dataframes to be paranoid
            pos_sim_ev_sorted = pos_sim_ev.sort_values(by='onset')
            dur_sim_ev_sorted = dur_sim_ev.sort_values(by='onset')

        # get a list of the event files with occurances of faces
        event_files = sorted(glob(eventdir + '/*'))
        assert len(event_files) == 8
        # get additional events from the location annotation
        location_annotation = pd.read_csv(annot_dir, sep='\t')

        # get all settings with more than one occurrence
        setting = [set for set in location_annotation.setting.unique()
                   if (location_annotation.setting[location_annotation.setting == set].value_counts()[0] > 1)]

        # get onsets and durations
        onset = []
        duration = []
        condition = []
        # lets also append an amplitude, we need this if multimatch is included
        # and should not hurt if its not included
        amplitude = []
        for set in setting:
            for i in range(location_annotation.setting[location_annotation['setting'] == set].value_counts()[0]):
                onset.append(location_annotation[location_annotation['setting'] == set]['onset'].values[i])
                duration.append(location_annotation[location_annotation['setting'] == set]['duration'].values[i])
            condition.append([set] * (i + 1))
            amplitude.append([1.0] * (i + 1))
        # flatten conditions and amplitudes
        condition = [y for x in condition for y in x]
        amplitude = [y for x in amplitude for y in x]
        assert len(condition) == len(onset) == len(duration) == len(amplitude)

        # concatenate the strings
        condition_str = [set.replace(' ', '_') for set in condition]
        condition_str = ['location_' + set for set in condition_str]

        # put it in a dataframe
        locations = pd.DataFrame({
            'onset': onset,
            'duration': duration,
            'condition': condition_str,
            'amplitude': amplitude
        })

        # sort according to onsets to be paranoid
        locations_sorted = locations.sort_values(by='onset')

        # this is a dataframe encoding flow of time
        time_forward = pd.DataFrame([{
            'condition': 'time+',
            'onset': location_annotation['onset'][i],
            'duration': 1.0,
            'amplitude': 1.0}
            for i in range(len(location_annotation) - 1)
            if location_annotation['flow_of_time'][i] in ['+', '++']])

        time_back = pd.DataFrame([{
            'condition': 'time-',
            'onset': location_annotation['onset'][i],
            'duration': 1.0,
            'amplitude': 1.0} for i in range(len(location_annotation) - 1)
            if location_annotation['flow_of_time'][i] in ['-', '--']])

        # sort according to onsets to be paranoid
        time_forward_sorted = time_forward.sort_values(by='onset')
        time_back_sorted = time_back.sort_values(by='onset')

        scene_change = pd.DataFrame([{
            'condition': 'scene-change',
            'onset': location_annotation['onset'][i],
            'duration': 1.0,
            'amplitude': 1.0}
            for i in range(len(location_annotation) - 1)])

        scene_change_sorted = scene_change.sort_values(by='onset')

        # this is a dataframe encoding exterior
        exterior = pd.DataFrame([{
            'condition': 'exterior',
            'onset': location_annotation['onset'][i],
            'duration': location_annotation['duration'][i],
            'amplitude': 1.0}
            for i in range(len(location_annotation) - 1)
            if (location_annotation['int_or_ext'][i] == 'ext')])

        # sort according to onsets to be paranoid
        exterior_sorted = exterior.sort_values(by='onset')

        # this is a dataframe encoding nighttime
        night = pd.DataFrame([{'condition': 'night',
                               'onset': location_annotation['onset'][i],
                               'duration': location_annotation['duration'][i],
                               'amplitude': 1.0}
                              for i in range(len(location_annotation) - 1)
                              if (location_annotation['time_of_day'][i] == 'night')])

        # sort according to onsets to be paranoid
        night_sorted = night.sort_values(by='onset')

        assert np.all(locations_sorted.onset[1:].values >= locations_sorted.onset[:-1].values)
        assert np.all(time_back_sorted.onset[1:].values >= time_back_sorted.onset[:-1].values)
        assert np.all(time_forward_sorted.onset[1:].values >= time_forward_sorted.onset[:-1].values)
        assert np.all(exterior_sorted.onset[1:].values >= exterior_sorted.onset[:-1].values)
        assert np.all(night_sorted.onset[1:].values >= night_sorted.onset[:-1].values)
        assert np.all(scene_change_sorted.onset[1:].values >= scene_change_sorted.onset[:-1].values)
        if multimatch:
            assert np.all(pos_sim_ev_sorted.onset[1:].values >= pos_sim_ev_sorted.onset[:-1].values)
            assert np.all(dur_sim_ev_sorted.onset[1:].values >= dur_sim_ev_sorted.onset[:-1].values)

        # initialize the list of dicts that gets later passed to the glm
        events_dicts = []
        # This is relevant to later stack all dataframes together
        # and paranoidly make sure that they have the same columns
        cols = ['onset', 'duration', 'condition', 'amplitude']

        for run in runs:
            # get face data
            eventfile = sorted(event_files)[run]
            events = pd.read_csv(eventfile, sep='\t')

            for index, row in events.iterrows():

                # disregard no faces, put everything else into event structure
                if row['condition'] != 'no_face':
                    dic = {
                        'onset': row['onset'] + runonsets[run],
                        'duration': row['duration'],
                        'condition': row['condition'],
                        'amplitude': 1.0
                    }
                    events_dicts.append(dic)

        # events for runs
        run_reg = pd.DataFrame([{
            'onset': runonsets[i],
            'duration': abs(runonsets[i] - runonsets[i + 1]),
            'condition': 'run-' + str(i + 1),
            'amplitude': 1.0}
            for i in range(7)])

        # get all of these wonderful dataframes into a list and squish them
        dfs = [locations_sorted[cols], scene_change_sorted[cols],
               time_back_sorted[cols], time_forward_sorted,
               exterior_sorted[cols], night_sorted[cols], run_reg[cols]]
        if multimatch:
            dfs.append(pos_sim_ev_sorted[cols])
            dfs.append(dur_sim_ev_sorted[cols])
        # lets also reset the index here
        allevents = pd.concat(dfs).reset_index()

        # save all non-face related events in an event file, just for the sake of it
        # allevents.to_csv(results_dir + '/' + 'non_face_regs.tsv', sep='\t', index=False)

        # append non-faceevents to event structure for glm
        for index, row in allevents.iterrows():
            dic = {
                'onset': row['onset'],
                'duration': row['duration'],
                'condition': row['condition'],
                'amplitude': row['amplitude']
            }
            events_dicts.append(dic)

       # # save this event dicts structure  as a tsv file
       # import csv
       # with open(results_dir + '/' + 'full_event_file.tsv', 'w') as tsvfile:
       #     fieldnames = ['onset', 'duration', 'condition', 'amplitude']
       #     writer = csv.DictWriter(tsvfile, fieldnames=fieldnames, delimiter='\t')
       #     writer.writeheader()
       #     writer.writerows(events_dicts)
       # # save this event file also as json file... can there ever be enough different files...
       # import json
       # with open(results_dir + '/' + 'allevents.json', 'w') as f:
       #     json.dump(events_dicts, f)

    # if we're doing the localizer dataset, our life is so much easier
    elif analysis == 'localizer':

        # average onsets into one event file
        events = get_group_events(eventdir)
       # # save the event_file
       # fmt = "%10.3f\t%10.3f\t%16s\t%60s"
       # np.savetxt(results_dir + 'group_events.tsv', events, delimiter='\t', comments='',
       #            header='onset\tduration\ttrial_type\tstim_file', fmt=fmt)

        # get events into dictionary
        events_dicts = []
        for i in range(0, len(events)):
            dic = {
                'onset': events[i][0],
                'duration': events[i][1],
                'condition': events[i][2],
                'amplitude': 1
            }
            events_dicts.append(dic)

    return events_dicts
