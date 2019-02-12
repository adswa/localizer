#!/usr/bin/env python

import mvpa2.suite as mv
import numpy as np
from glob import glob
import pandas as pd
import os

"""
One script to rule them all:
This script shall be able to handle all analysis.
We will all be thanking datalad run and rerun once the analysis are
older than a week, for noone will remember the army of commandline
arguments specified. So in advance: Thanks, Kyle!

Command line specifications are as follows:
    --inputfile:    str, a transposed group dataset
    --output:       Str, absolute (!) path, the directory where 
                    results should go (will be created if it does not exist)
    --classifier:   Str, The classifier of choice
    --bilateral:    Boolean, Option on whether to combine ROIs of the 
                    hemispheres (True, default) or have them separate
    --dataset:      Str, Option to use the full dataset or on a dataset 
                    with only ROIs
    --coords:       Str, Option to include coordinates into the dataset
                    or run the analysis only on coordinates
    --niceplot:     Boolean, option to plot a pretty confusion matrix with
                    matplotlib
    --glm: Boolean, Option to specify whether a glm of sensitivities regressed 
                    onto stimulation description should be computed.
    IF --glm True, specify:
        --eventdir:         Where does the script find the necessary event 
                            files for derivation of regressors?
        --multimatch:       Where are the mean multimatch files (per runs), if the
                            should be included?(i.e.
                            sourcedata/multimatch/output/run_*/means.csv)
        --plot_time_series: Boolean, should a time series plot of the 
                            sensitivity and glm fit be produced?
        IF --plot_time_series True:
                --roipair:      two ROIs for which the glm time series will be plotted
                --analysis:     Is the glm run on localizer or avmovie data?
            IF --analysis 'avmovie'
                    --include_all_regressors:   Boolean, should all regressors 
                                                be put into the timeseries plot?
                    --annotation:               str, path to the singular, 
                                                long researchcut movie annotation
                    --multimatch:               Path to allruns.tsv multimatch
                                                results. Uses Position and
                                                Duration Similarity.
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
    mv.h5save(results_dir + 'ds_ROIs.hdf5', ds_ROIs)
    print('Combined lateralized ROIs for the provided dataset and saved the dataset.')
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
                   ACC=None):
    """ This function plots the classification results as a confusion matrix.
    Specify ACC as cv.ca.stats.stats['mean(ACC)'] to display accuracy in the
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


def dotheclassification(ds,
                        classifier,
                        bilateral,
                        ds_type,
                        store_sens=True):
    """ Dotheclassification does the classification.
    Input: the dataset on which to perform a leave-one-out crossvalidation with a classifier
    of choice.
    Specify: the classifier to be used (gnb (linear gnb), l-sgd (linear sgd), sgd)
             whether the sensitivities should be computed and stored for later use
             whether the dataset has ROIs combined across hemisphere (bilateral)
    """
    import matplotlib.pyplot as plt

    if classifier == 'gnb':

        # set up classifier
        prior = 'ratio'
        if bilateral:
            targets = 'bilat_ROIs'
        else:
            targets = 'all_ROIs'

        clf = mv.GNB(common_variance=True,
                 prior=prior,
                 space=targets)

    elif classifier == 'sgd':

        # set up the dataset: If I understand the sourcecode correctly, the
        # SGDclassifier wants to have unique labels in a sample attribute
        # called 'targets' and is quite stubborn with this name - I could not convince
        # it to look for targets somewhere else, so now I'm catering to his demands
        if bilateral:
            ds.sa['targets'] = ds.sa.bilat_ROIs
        else:
            ds.sa['targets'] = ds.sa.all_ROIs

        # necessary I believe regardless of the SKLLearnerAdapter
        from sklearn.linear_model import SGDClassifier


        clf = mv.SKLLearnerAdapter(SGDClassifier(loss='hinge',
                                                 penalty='l2',
                                                 class_weight='balanced'))

    elif classifier == 'l-sgd':
        # set up the dataset: If I understand the sourcecode correctly, the
        # Stochastic Gradient Descent wants to have unique labels in a sample attribute
        # called 'targets' and is quite stubborn with this name - I could not convince
        # it to look for targets somewhere else, so now I catering to his demands
        if bilateral:
            ds.sa['targets'] = ds.sa.bilat_ROIs
        else:
            ds.sa['targets'] = ds.sa.all_ROIs

        # necessary I believe regardless of the SKLLearnerAdapter
        from sklearn.linear_model import SGDClassifier

        # get a stochastic gradient descent into pymvpa by using the SKLLearnerAdapter.
        # Get it to perform 1 vs 1 decisions (instead of one vs all) with the MulticlassClassifier
        clf = mv.MulticlassClassifier(mv.SKLLearnerAdapter(SGDClassifier(loss='hinge',
                                                                         penalty='l2',
                                                                         class_weight='balanced'
                                                                         )))

    # prepare for callback of sensitivity extraction within CrossValidation
    sensitivities = []
    if store_sens:
        def store_sens(data, node, result):
            sens = node.measure.get_sensitivity_analyzer(force_train=False)(data)
            # we also need to manually append the time attributes to the sens ds
            sens.fa['time_coords'] = data.fa['time_coords']
            sens.fa['chunks'] = data.fa['chunks']
            sensitivities.append(sens)

        # do a crossvalidation classification and store sensitivities
        cv = mv.CrossValidation(clf, mv.NFoldPartitioner(attr='participant'),
                                errorfx=mv.mean_match_accuracy,
                                enable_ca=['stats'],
                                callback=store_sens)
    else:
        # don't store sensitivities
        cv = mv.CrossValidation(clf, mv.NFoldPartitioner(attr='participant'),
                                errorfx=mv.mean_match_accuracy,
                                enable_ca=['stats'])
    results = cv(ds)
    # save classification results

    with open(results_dir + 'CV_results.txt', 'a') as f:
        f.write(cv.ca.stats.as_string(description=True))

    # printing of the confusion matrix
    # first, get the labels according to the size of dataset. This is in principle
    # superflous (get_desired_labels() would exclude brain if it wasn't in the data),
    # but it'll make sure that a permitted ds_type was specified.
    if ds_type == 'full':
        if bilateral:
            desired_order = ['brain', 'VIS', 'LOC', 'OFA', 'FFA', 'EBA', 'PPA']
            if 'FEF' in ds.sa.bilat_ROIs:
                desired_order.append('FEF')
        else:
            desired_order = ['brain', 'VIS', 'left LOC', 'right LOC',
                             'left OFA', 'right OFA', 'left FFA',
                             'right FFA', 'left EBA', 'right EBA',
                             'left PPA', 'right PPA']
            if 'FEF' in ds.sa.all_ROIs:
                desired_order.extend(['right FEF', 'left FEF'])
    if ds_type == 'stripped':
        if bilateral:
            desired_order = ['VIS', 'LOC', 'OFA', 'FFA', 'EBA', 'PPA']
            if 'FEF' in ds.sa.bilat_ROIs:
                desired_order.append('FEF')
        else:
            desired_order = ['VIS', 'left LOC', 'right LOC',
                             'left OFA', 'right OFA', 'left FFA',
                             'right FFA', 'left EBA', 'right EBA',
                             'left PPA', 'right PPA']
            if 'FEF' in ds.sa.all_ROIs:
                desired_order.extend(['right FEF', 'left FEF'])

    labels = get_known_labels(desired_order,
                              cv.ca.stats.labels)

    # plot the confusion matrix with pymvpas build-in plot function currently fails
    cv.ca.stats.plot(labels=labels,
                     numbers=True,
                     cmap='gist_heat_r')
    plt.savefig(results_dir + 'CV_confusion_matrix.png')
    if niceplot:
        ACC = cv.ca.stats.stats['mean(ACC)']
        plot_confusion(cv,
                       labels,
                       fn=results_dir + 'CV_confusion_matrix.svg',
                       figsize=(9, 9),
                       vmax=100,
                       cmap='Blues',
                       ACC='%.2f' % ACC)
    mv.h5save(results_dir + 'cv_classification_results.hdf5', results)
    print('Saved the crossvalidation results.')
    if store_sens:
        mv.h5save(results_dir + 'sensitivities_nfold.hdf5', sensitivities)
        print('Saved the sensitivities.')
    # results now has the overall accuracy. results.samples gives the
    # accuracy per participant.
    # sensitivities contains a dataset for each participant with the
    # sensitivities as samples and class-pairings as attributes
    #import pdb; pdb.set_trace()
    return sensitivities, cv


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


def dotheglm(sensitivities,
             eventdir,
             normalize,
             analysis,
             classifier,
             multimatch,
             annot_dir=None):

    """dotheglm() regresses sensitivities obtained during
    cross validation onto a functional description of the
    paradigm.
    If specified with normalize = True, sensitivities
    are normed to their L2 norm.
    The the sensitivities will be vstacked into one
    dataset according to which classifier was used, and
    how large the underlying dataset was.
    The average sensitivity per roi pair will be calculated
    with the mean_group_sample() function.
    The resulting averaged sensitivity file will be transposed
    with a TransposeMapper().
    According to which analysis is run, the appropriate event
     and if necessary annotation files
    will be retrieved and read into the necessary data structure.
    """
    if normalize == True:
        mean_sens = norm_and_mean(norm=True,
                                  bilateral=bilateral,
                                  classifier=classifier,
                                  sensitivities=sensitivities
                                  )

        #import pdb; pdb.set_trace()
    elif normalize == False:
        mean_sens = norm_and_mean(norm=False,
                                  bilateral=bilateral,
                                  classifier=classifier,
                                  sensitivities=sensitivities
                                  )
    # transpose the averaged sensitivity dataset
    mean_sens_transposed = mean_sens.get_mapped(mv.TransposeMapper())
    # if we're analyzing the avmovie data:
    if analysis == 'avmovie':

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
                data = pd.read_csv(multimatch_file, sep = '\t')
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

        # check whether chunks are increasing as well as sanity check
        chunks = mean_sens_transposed.sa.chunks
        assert np.all(chunks[1:] >= chunks[:-1])

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
        allevents.to_csv(results_dir + '/' + 'non_face_regs.tsv', sep='\t', index=False)

        # append non-faceevents to event structure for glm
        for index, row in allevents.iterrows():
            dic = {
                'onset': row['onset'],
                'duration': row['duration'],
                'condition': row['condition'],
                'amplitude': row['amplitude']
            }
            events_dicts.append(dic)

        # save this event dicts structure  as a tsv file
        import csv
        with open(results_dir + '/' + 'full_event_file.tsv', 'w') as tsvfile:
            fieldnames = ['onset', 'duration', 'condition', 'amplitude']
            writer = csv.DictWriter(tsvfile, fieldnames=fieldnames, delimiter='\t')
            writer.writeheader()
            writer.writerows(events_dicts)
        # save this event file also as json file... can there ever be enough different files...
        import json
        with open(results_dir + '/' + 'allevents.json', 'w') as f:
            json.dump(events_dicts, f)

    # if we're doing the localizer dataset, our life is so much easier
    elif analysis == 'localizer':

        # average onsets into one event file
        events = get_group_events(eventdir)
        # save the event_file
        fmt = "%10.3f\t%10.3f\t%16s\t%60s"
        np.savetxt(results_dir + 'group_events.tsv', events, delimiter='\t', comments='',
                   header='onset\tduration\ttrial_type\tstim_file', fmt=fmt)

        # get events into dictionary
        events_dicts = []
        for i in range(0, len(events)):
            dic = {
                'onset': events[i][0],
                'duration': events[i][1],
                'condition': events[i][2],
                'amplitude': events[i][3]
            }
            events_dicts.append(dic)

    # do the glm - we've earned it
    hrf_estimates = mv.fit_event_hrf_model(mean_sens_transposed,
                                           events_dicts,
                                           time_attr='time_coords',
                                           condition_attr='condition',
                                           design_kwargs=dict(drift_model='blank'),
                                           glmfit_kwargs=dict(model='ols'),
                                           return_model=True)

    mv.h5save(results_dir + '/' + 'sens_glm_results.hdf5', hrf_estimates)
    print('calculated the glm, saving results.')

    return hrf_estimates


def makeaplot_localizer(events,
                        sensitivities,
                        hrf_estimates,
                        roi_pair,
                        normalize,
                        classifier,
                        bilateral,
                        fn=True,
                        ):
    """
    This produces a time series plot for the roi class comparison specified in
    roi_pair such as roi_pair = ['left FFA', 'left PPA'] for the localizer data.
    """
    import matplotlib.pyplot as plt

    if normalize:
        mean_sens = norm_and_mean(norm=True,
                                  bilateral=bilateral,
                                  classifier=classifier,
                                  sensitivities=sensitivities
                                  )
    else:
        mean_sens = norm_and_mean(norm=False,
                                  bilateral=bilateral,
                                  classifier=classifier,
                                  sensitivities=sensitivities
                                  )
    # transpose the averaged sensitivity dataset
    mean_sens_transposed = mean_sens.get_mapped(mv.TransposeMapper())

    # some parameters
    # get the conditions, and reorder them into a nice order
    block_design = sorted(np.unique(events['trial_type']))
    reorder = [0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11]
    block_design = [block_design[i] for i in reorder]

    # end indices to chunk timeseries into runs
    run_startidx = np.array([0, 157, 313, 469])
    run_endidx = np.array([156, 312, 468, 624])

    runs = np.unique(mean_sens_transposed.sa.chunks)

    roi_pair_idx = get_roi_pair_idx(bilateral,
                                    classifier,
                                    roi_pair,
                                    hrf_estimates)
    roi_betas_ds = hrf_estimates[:, roi_pair_idx]
    roi_sens_ds = mean_sens_transposed[:, roi_pair_idx]
    for run in runs:
        fig, ax = plt.subplots(1, 1, figsize=[18, 10])
        colors = ['#7b241c', '#e74c3c', '#154360', '#3498db', '#145a32', '#27ae60',
                  '#9a7d0a', '#f4d03f', '#5b2c6f', '#a569bd', '#616a6b', '#ccd1d1']
        plt.suptitle('Timecourse of sensitivities, {} versus {}, run {}'.format(roi_pair[0],
                                                                                roi_pair[1],
                                                                                run + 1),
                     fontsize='large')
        plt.xlim([0, max(mean_sens_transposed.sa.time_coords)])
        plt.ylim([-5, 7])
        plt.xlabel('Time in sec')
        plt.legend(loc=1)
        plt.grid(True)
        # for each stimulus, plot a color band on top of the plot
        for stimulus in block_design:
            onsets = events[events['trial_type'] == stimulus]['onset'].values
            durations = events[events['trial_type'] == stimulus]['duration'].values
            stimulation_end = np.sum([onsets, durations], axis=0)
            r_height = 1
            color = colors[0]
            y = 6

            # get the beta corresponding to the stimulus to later use in label

            for i in range(len(onsets)):
                beta = roi_betas_ds.samples[hrf_estimates.sa.condition == stimulus.replace(" ", ""), 0]
                r_width = durations[i]
                x = stimulation_end[i]
                rectangle = plt.Rectangle((x, y),
                                          r_width,
                                          r_height,
                                          fc=color,
                                          alpha=0.5,
                                          label='_'*i +
                                                stimulus.replace(" ", "") +
                                                '(' + str('%.2f' % beta) + ')')
                plt.gca().add_patch(rectangle)
                plt.legend(loc=1)
            del colors[0]

        times = roi_sens_ds.sa.time_coords[run_startidx[run]:run_endidx[run]]

        ax.plot(times, roi_sens_ds.samples[run_startidx[run]:run_endidx[run]], '-', color='black', lw=1.0)
        glm_model = hrf_estimates.a.model.results_[0.0].predicted[run_startidx[run]:run_endidx[run], roi_pair_idx]
        ax.plot(times, glm_model, '-', color='#7b241c', lw=1.0)
        model_fit = hrf_estimates.a.model.results_[0.0].R2[roi_pair_idx]
        plt.title('R squared: %.2f' % model_fit)
        if fn:
            plt.savefig(results_dir +
                        'timecourse_localizer_glm_sens_{}_vs_{}_run-{}.svg'.format(roi_pair[0],
                                                                                   roi_pair[1],
                                                                                   run + 1))


def makeaplot_avmovie(events,
                      sensitivities,
                      hrf_estimates,
                      roi_pair,
                      normalize,
                      bilateral,
                      classifier,
                      fn=None,
                      include_all_regressors=False):
    """
    This produces a time series plot for the roi class comparison specified in
    roi_pair such as roi_pair = ['left FFA', 'left PPA'].
    If include_all_regressors = True, the function will create a potentially overloaded
    legend with all of the regressors, regardless of they occurred in the run. (Plotting
    then takes longer, but is a useful option if all regressors are of relevance and can
    be twitched in inkscape).
    If the figure should be saved, spcify an existing path in the parameter fn.

    # TODO's for the future: runs=None, overlap=False, grouping (should be a way to not rely
    # on hardcoded stimuli and colors within function anymore, with Ordered Dicts):

    """
    import matplotlib.pyplot as plt

    if normalize:
        mean_sens = norm_and_mean(norm=True,
                                  bilateral=bilateral,
                                  classifier=classifier,
                                  sensitivities=sensitivities
                                  )
    else:
        mean_sens = norm_and_mean(norm=False,
                                  bilateral=bilateral,
                                  classifier=classifier,
                                  sensitivities=sensitivities
                                  )
    # transpose the averaged sensitivity dataset
    mean_sens_transposed = mean_sens.get_mapped(mv.TransposeMapper())

    chunks = mean_sens_transposed.sa.chunks
    assert np.all(chunks[1:] >= chunks[:-1])

    # TR was not preserved/carried through in .a
    # so we will guestimate it based on the values of time_coords
    runs = np.unique(mean_sens_transposed.sa.chunks)
    tc = mean_sens_transposed.sa.time_coords
    TRdirty = sorted(np.unique(tc[1:] - tc[:-1]))[-1]
    assert np.abs(np.round(TRdirty, decimals=2) - TRdirty) < 0.0001

    mean_sens_transposed.sa.time_coords = np.arange(len(mean_sens_transposed)) * TRdirty
    # those
    runlengths = [np.max(tc[mean_sens_transposed.sa.chunks == run]) + TRdirty
                  for run in runs]
    runonsets = [sum(runlengths[:run]) for run in runs]
    # just append any large number to accomodate the fact that the last run also needs an
    # at some point.
    runonsets.append(99999)

    roi_pair_idx = get_roi_pair_idx(bilateral,
                                    classifier,
                                    roi_pair,
                                    hrf_estimates)

    roi_betas_ds = hrf_estimates[:, roi_pair_idx]
    roi_sens_ds = mean_sens_transposed[:, roi_pair_idx]
    from collections import OrderedDict
    block_design_betas = OrderedDict(
        sorted(zip(roi_betas_ds.sa.condition, roi_betas_ds.samples[:, 0]),
               key=lambda x: x[1]))
    block_design = list(block_design_betas)
    for run in runs:
        fig, ax = plt.subplots(1, 1, figsize=[18, 10])
        colors = ['#7b241c', '#e74c3c', '#154360', '#3498db', '#145a32', '#27ae60',
                  '#9a7d0a', '#f4d03f', '#5b2c6f', '#a569bd', '#616a6b', '#ccd1d1']
        plt.suptitle('Timecourse of sensitivities, {} versus {}, run {}'.format(roi_pair[0],
                                                                                roi_pair[1],
                                                                                run + 1),
                     fontsize='large')
        # 2 is a TR here... sorry, we are in rush
        run_onset = int(runonsets[run] // 2)
        run_offset = int(runonsets[run + 1] // 2)
        # for each run, adjust the x-axis
        plt.xlim([min(mean_sens_transposed.sa.time_coords[run_onset:int(run_offset)]),
                  max(mean_sens_transposed.sa.time_coords[run_onset:int(run_offset)])])
        plt.ylim([-2.7, 4.5])
        plt.xlabel('Time in sec')
        plt.legend(loc=1)
        plt.grid(True)

        # for each stimulus, plot a color band on top of the plot
        for stimulus in block_design:
            color = colors[0]
            print(stimulus)
            condition_event_mask = events['condition'] == stimulus
            onsets = events[condition_event_mask]['onset'].values
            onsets_run = [time for time in onsets
                          if np.logical_and(time > run_onset * 2, time < run_offset * 2)]
            durations = events[condition_event_mask]['duration'].values
            durations_run = [dur for idx, dur in enumerate(durations)
                             if np.logical_and(onsets[idx] > run_onset * 2,
                                               onsets[idx] < run_offset * 2)]
            # prepare for plotting
            r_height = 0.3
            y = 4
            if stimulus.startswith('run'):
                continue
            if stimulus.startswith('location'):
                # gradually decrease alpha level over occurances of location stims
                y -= r_height
                color = 'darkgreen'
            elif 'face' in stimulus:
                if stimulus == 'many_faces':
                    color = 'tomato'
                else:
                    color = 'firebrick'
            elif stimulus == 'exterior':
                color = 'cornflowerblue'
                y -= 2 * r_height
            elif stimulus.startswith('time'):
                color = 'darkslategrey'
                y -= 3 * r_height
            elif stimulus == 'night':
                color = 'slategray'
                y -= 4 * r_height
            elif stimulus == 'scene-change':
                color = 'black'
                y -= 5 * r_height

            # get the beta corresponding to the stimulus to later use in label
            beta = roi_betas_ds.samples[hrf_estimates.sa.condition == stimulus, 0]

            if include_all_regressors and onsets_run == []:
                # if there are no onsets for a particular regressor,
                # but we want to print all
                # regressors, set i manually to 0
                rectangle = plt.Rectangle((0, 0),
                                          0,
                                          0,
                                          fc=color,
                                          alpha=0.5,
                                          label='_' * 0 \
                                                + stimulus.replace(" ", "") +
                                                '(' + str('%.2f' % beta) + ')')
                plt.gca().add_patch(rectangle)

            for i, x in enumerate(onsets_run):
                # We need the i to trick the labeling. It will
                # attempt to plot every single occurance
                # of a stimulus with numbered labels. However,
                # appending a '_' to the label makes
                # matplotlib disregard it. If we attach an '_' * i
                # to the label, all but the first onset
                # get a '_' prefix and are ignored.
                r_width = durations_run[i]
                rectangle = plt.Rectangle((x, y),
                                          r_width,
                                          r_height,
                                          fc=color,
                                          alpha=0.5,
                                          label='_' * i + \
                                                stimulus.replace(" ", "") +
                                                '(' + str('%.2f' % beta) + ')')
                plt.gca().add_patch(rectangle)
                plt.legend(loc=1)
                # plt.axis('scaled')
                # del colors[0]

        times = roi_sens_ds.sa.time_coords[run_onset:run_offset]

        ax.plot(times, roi_sens_ds.samples[run_onset:run_offset], '-', color='black', lw=1.0)
        # plot glm model results
        glm_model = hrf_estimates.a.model.results_[0.0].predicted[run_onset:int(run_offset), roi_pair_idx]
        ax.plot(times, glm_model, '-', color='#7b241c', lw=1.0)
        model_fit = hrf_estimates.a.model.results_[0.0].R2[roi_pair_idx]
        plt.title('R squared: %.2f' % model_fit)
        if fn:
            plt.savefig(results_dir +
                        'timecourse_avmovie_glm_sens_{}_vs_{}_run-{}.svg'.format(roi_pair[0],
                                                                                 roi_pair[1],
                                                                                 run + 1))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--inputfile', help="An hdf5 file of the avmovie "
                        "data with functional ROI information, transposed",
                        required=True)
    parser.add_argument('--analysis', help="[If glm is computed:] Which dataset is "
                        "the analysis based on, 'localizer' or 'avmovie'", type=str)
    parser.add_argument('-a', '--annotation', help="Input a single, full movie"
                        "spanning location annotation file, if you want to compute"
                        "the glm.")
    parser.add_argument('-e', '--eventdir', help="Input the directory name under which" 
                        "the downsamples, run-wise event files can be found, if you " 
                        "want to compute the glm.")
    parser.add_argument('-bi', '--bilateral', help="If false, computation will "
                        "be made on hemisphere-specific ROIs (i.e. left FFA, "
                        "right FFA", default=True)
    parser.add_argument('-g', '--glm', help="Should a glm on the sensitivities be"
                        "computed? Defaults to True, as long as the classification"
                        "isn't done on an only-coordinates dataset (as specified "
                        "with the --coords flag)", default=True)
    parser.add_argument('-ds', '--dataset', help="Specify whether the analysis \
                        should be done on the full dataset or on the dataset \
                        with only ROIs: 'full' or 'stripped' (default: stripped)",
                        type=str, default='stripped')
    parser.add_argument('-c', '--coords', help="Should coordinates be included in \
                        the dataset? ('with-coordinates').Should a sanity check \
                        with only coordinates without fmri data be performed? \
                        ('only-coordinates'). Should coordinates be disregard? \
                        ('no-coordinates') Default: 'no-coordinates'.", type=str,
                        default='no-coordinates')
    parser.add_argument('-o', '--output', help="Please specify an output directory"
                        "name (absolute path) to store the analysis results", type=str)
    parser.add_argument('-r', '--roipair', nargs='+', help="Specify two ROIs for which "
                        "the glm timecourse should be plotted. Default for now is right"
                        " FFA & right PPA in lateralized dataset, FFA & PPA in bilateral"
                        " dataset. Specify as --roipair 'FFA' 'PPA'")
    parser.add_argument('-n', '--niceplot', help="If true, the confusion matrix of the "
                        "classification will be plotted with Matplotlib instead of build "
                        "in functions of pymvpa.", default=False)
    parser.add_argument('-ps', '--plot_time_series', help="If True, the results of the "
                        "glm will be plotted as a timeseries per run.", default=False)
    parser.add_argument('-ar', '--include_all_regressors', help="If you are plotting the "
                        "time series, do you want the plot to contain all of the"
                        " regressors?", default=False)
    parser.add_argument('--classifier', help="Which classifier do you want to use? Options:"
                        "linear Gaussian Naive Bayes ('gnb'), linear (binary) stochastic "
                        "gradient descent (l-sgd), stochastic gradient descent (sgd)",
                        type=str, required=True)
    parser.add_argument('--normalize', help="Should the sensitivities used for the glm be"
                        "normalized by their L2 norm? True/False",
                        default=True)
    parser.add_argument('--multimatch', help="path to multimatch mean results"
                        "per run. If given, the similarity measures"
                        "for position and duration will be included in the"
                        "avmovie glm analysis. Provide path including file name,"
                        "as in 'sourcedata/multimatch/output/run_*/means.tsv'")

    ## TODO: REMODNAV AND MULTIMATCH DATA as additional events

    args = parser.parse_args()

    # get the data
    ds_file = args.inputfile
    ds = mv.h5load(ds_file)

    # prepare the output path
    results_dir = '/' + args.output + '/'
    # create the output dir if it doesn't exist
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    # get more information about what is being calculated
    ds_type = args.dataset                      # stripped --> no brain, no overlap,
                                                #  full --> no overlap
    if args.glm == 'True' or args.glm == True:
        glm = True
    elif args.glm == 'False' or args.glm == False:
        glm = False                             # True or False
    if args.bilateral == 'True' or args.bilateral == True:
        bilateral = True
    elif args.bilateral == 'False' or args.bilateral == False:
        bilateral = False                       # True or False
    if args.niceplot == 'True' or args.niceplot == True:
        niceplot = True
    elif args.niceplot == 'False' or args.niceplot == False:
        niceplot = False                        # False or True
    if args.normalize == 'True' or args.normalize == True:
        normalize = True
    elif args.normalize == 'False' or args.normalize == False:
        normalize = False                       # True or False
    if args.plot_time_series == 'True' or args.plot_time_series == True:
        plot_ts = True
    elif args.plot_time_series == 'False' or args.plot_time_series == False:
        plot_ts = False                         # False or True
    incl_regs = args.include_all_regressors
    coords = args.coords                        # no-coords --> leave ds as is,
                                                # with-coords --> incl. coords,
                                                # only-coords --> only coords
    classifier = args.classifier                # gnb, sgd, l-sgd --> multiclassclassifier

    # fail early, if classifier is not appropriately specified.
    allowed_clfs = ['sgd', 'l-sgd', 'gnb']
    if classifier not in allowed_clfs:
        raise ValueError("The classifier of choice must be one of {},"
                         " however, {} was specified.".format(allowed_clfs,
                                                                classifier))

    # fail early, if ds_type is not appropriately specified.
    allowed_ds_types = ['stripped', 'full']
    if ds_type not in allowed_ds_types:
        raise ValueError("The ds_type of choice must be "
                         "one of {}, however, {} was specified.".format(allowed_ds_types,
                                                                        ds_type))

    # the default is to store sensitivities during classification
    # (TODO: implement sens callback in SGD all-vs-1)
    store_sens = True

    # get the data into the appropriate shape.
    # If the dataset should be stripped, apply
    # 'full' stripping. If not, apply only 'sparse' stripping
    # that would exclude any overlap in the data.
    if ds_type == 'stripped':
        ds = strip_ds(ds, order='full')
    else:
        ds = strip_ds(ds, order='sparse')

    # combine ROIs of the hemispheres
    if bilateral:
        ds = bilateralize(ds)

    # append coordinates if specified
    if coords == 'with-coordinates':
        ds = get_voxel_coords(ds,
                              append=True,
                              zscore=True)
        store_sens = False
        glm = False
        print('The classification will be done with coordinates.'
              'Note: no sequential glm analysis on sensitivities'
              ' will be done if this option is specified.')
    # or append coordinates and get rid of fmri data is specified
    elif coords == 'only-coordinates':
        ds = get_voxel_coords(ds,
                              append=False,
                              zscore=False)
        # if there is no fmri data in the ds, don't attempt to
        # get sensitivities and only to a classification
        store_sens = False
        glm = False
        print('The classification will be done with coordinates only '
              '(are you doing a sanity check?).'
              'Note: no sequential glm analysis on sensitivities'
              ' will be done if this option is specified.')

    # if we are running a glm, do I have everything I need for the computation?
    if glm:
        # which dataset am I being run on?
        if args.analysis:
            analysis = args.analysis
        else:
            print("You have specified to run a glm, however you have"
                  " not specified which dataset (avmovie/localizer) "
                  "the analysis is based on. Without this information"
                  " this script is not able to compute the glm.")

        # if the data basis is avmovie...
        if analysis == 'avmovie':
            print("The analysis will include a glm. Specified "
                  "input data (--analysis) is avmovie.")
            # are there glm inputs?
            if args.eventdir:
                eventdir = args.eventdir
                print("I received the following specification to find"
                      " event files for glm computation on avmovie data on"
                      "{}. Please check whether this looks correct to you."
                      " If I receive the *wrong* event files, results "
                      "will be weird.".format(eventdir))
            else:
                print("You have specified to run a glm, and that the data"
                      " basis you supplied is the data from the avmovie task."
                      "However, you did not specify a directory where to "
                      "find event files in under --eventdir")
            if args.annotation:
                annot_dir = args.annotation
            else:
                print("You have specified to run a glm, and that the data"
                      " basis you supplied is the data from the avmovie task."
                      "However, you did not specify a directory where to find"
                      " the single annotation file under --annotation")
            if args.multimatch:
                multimatch = args.multimatch
                print("Multimatch data will be included.")
            else:
                multimatch = False
                print("Multimatch data is not used.")

        #if the data basis is localizer...
        if analysis == 'localizer':
            print("The analysis will include a glm. Specified input "
                  "data (--analysis) is localizer.")
            # are there glm inputs?
            if args.eventdir:
                eventdir = args.eventdir
                print("I received the following specification to find event"
                      " files for glm computation on localizer data on"
                      "{}. Please check whether this looks correct to you."
                      " If I receive the *wrong* event files, results "
                      "will be weird.".format(eventdir))
                # fail early if there are no eventfiles:
                event_files = sorted(glob(eventdir + '*_events.tsv'))
                if len(event_files) == 0:
                    raise ValueError('No event files were discovered at the'
                                     ' specified location. Make sure you only'
                                     ' specify the directory the eventfiles '
                                     'are in, and not the names of the eventfiles.'
                                     ' The way the event files are globbed'
                                     ' is glob(eventdir + "*_events.tsv").')
            else:
                print("You have specified to run a glm, and that the data"
                      " basis you supplied is the data from the localizer task."
                      "However, you did not specify a directory where to "
                      "find event files in under --eventdir")

        # give feedback about which plots are made or not made.
        if plot_ts:
            print("The resulting time series plot will be produced.")

            if incl_regs:
                print("The time series plots will contain all regressors per plot.")
            else:
                print(
                    "The time series plots will only contain the "
                    "regressors that actually occurred in the respective run")
        else:
            print("The resulting time series plot will NOT be produced,"
                  " only the hrf estimates are saved.")

        if args.roipair:
            roi_pair = [i for i in args.roipair]
            if len(roi_pair) != 2:
                print('I expected exactly 2 ROIs for a comparison, specified as string'
                      'such as in --roipair "FFA" "PPA". However, I got {}. '
                      'I will default to plotting a comparison between '
                      '(right) FFA and PPA.'.format(args.roipair))
                if bilateral:
                    roi_pair = ['FFA', 'PPA']
                else:
                    roi_pair = ['right FFA', 'right PPA']
        print("This ROI pair is going to be used: {}".format(roi_pair))

        ## TODO: what happens to roi pair in the event of a sgd classifier 1-vs-all?
    # currently, related to the todo, the glm computation won't work in sgd
    # 1-vs-all classification as we can't derive the comparison roi yet. For now,
    # I will disable glm computation for these cases.
    if classifier == 'sgd':
        glm = False
        print("Currently, the glm computation won't work in sgd 1-vs-all \
            classification as we can't derive the comparison roi yet. For now,the \
            glm computation for these cases is disabled")

    sensitivities, cv = dotheclassification(ds,
                                            classifier=classifier,
                                            bilateral=bilateral,
                                            ds_type=ds_type,
                                            store_sens=store_sens)
    if (glm) and (analysis == 'avmovie'):
        hrf_estimates = dotheglm(sensitivities,
                                 normalize=normalize,
                                 classifier=classifier,
                                 analysis=analysis,
                                 annot_dir=annot_dir,
                                 eventdir=eventdir,
                                 multimatch=multimatch)
        if plot_ts:
            events = pd.read_csv(results_dir + 'full_event_file.tsv', sep='\t')
            makeaplot_avmovie(events,
                              sensitivities,
                              hrf_estimates,
                              roi_pair,
                              normalize=normalize,
                              classifier=classifier,
                              bilateral=bilateral,
                              fn=results_dir,
                              include_all_regressors=incl_regs)
    elif (glm) and (analysis == 'localizer'):
        hrf_estimates = dotheglm(sensitivities,
                                 normalize=normalize,
                                 analysis=analysis,
                                 classifier=classifier,
                                 eventdir=eventdir,
                                 multimatch=multimatch)
        if plot_ts:
            # read the event files, they've been produced by the glm
            events = pd.read_csv(results_dir + 'group_events.tsv',
                                 sep='\t')
            makeaplot_localizer(events,
                                sensitivities,
                                hrf_estimates,
                                roi_pair,
                                normalize=normalize,
                                classifier=classifier,
                                bilateral=bilateral,
                                fn=results_dir)


