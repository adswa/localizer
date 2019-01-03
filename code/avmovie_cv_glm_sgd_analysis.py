#!/home/adina/wtf/bin/python

import numpy as np
import mvpa2.suite as mv
from glob import glob
import pandas as pd
import os

"""
This script takes an existing transposed hdf5 dataset, containing the studyforrest
movie data and the data of the functional ROIs, as its input file.

To call the dataset, specify the following command line options/inputs:
BLABLABLA

This script uses a support vector machine as classifying algorithm of choice.
"""


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


def get_known_labels(desired_order, known_labels):
    """ Helper function to reorder ROI labels in a confusion matrix."""
    return [
        label
        for label in desired_order
        if label in known_labels
    ]


def bilateralize(ds):
    """combine lateralized ROIs in a dataset."""
    ds_ROIs = ds.copy('deep')
    ds_ROIs.sa['bilat_ROIs'] = [label.split(' ')[-1] for label in ds_ROIs.sa.all_ROIs]
    mv.h5save(results_dir + 'ds_ROIs.hdf5', ds_ROIs)
    print('Combined lateralized ROIs for the provided dataset and saved the dataset.')
    return ds_ROIs


# to print  a confusion matrix
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


def dotheclassification(ds,
                        bilateral,
                        store_sens=True):
    """ Dotheclassification does the classification. It builds a
    linear gaussian naive bayes classifier, performs a leave-one-out
    crossvalidation and stores the sensitivities from the SGD classifier of each
    fold in a combined dataset for further use in a glm.
    If sens == False, the sensitivities are not stored, and only a
    classification is performed"""
    import matplotlib.pyplot as plt
    # set up the dataset: If I understand the sourcecode correctly, the
    # MulticlassClassifier wants to have unique labels in a sample attribute
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

            # do a crossvalidation classification

        cv = mv.CrossValidation(clf, mv.NFoldPartitioner(attr='participant'),
                                errorfx=mv.mean_match_accuracy,
                                enable_ca=['stats'],
                                callback=store_sens)
    else:
        cv = mv.CrossValidation(clf, mv.NFoldPartitioner(attr='participant'),
                                errorfx=mv.mean_match_accuracy,
                                enable_ca=['stats'])
    results = cv(ds)
    # save classification results

    with open(results_dir + 'avmovie_clf.txt', 'a') as f:
        f.write(cv.ca.stats.as_string(description=True))
    # printing of the confusion matrix
    if bilateral:
        desired_order = ['VIS', 'LOC', 'OFA', 'FFA', 'EBA', 'PPA']
    else:
        desired_order = ['brain', 'VIS', 'left LOC', 'right LOC',
                         'left OFA', 'right OFA', 'left FFA',
                         'right FFA', 'left EBA', 'right EBA',
                         'left PPA', 'right PPA']
    labels = get_known_labels(desired_order,
                              cv.ca.stats.labels)

    # plot the confusion matrix with pymvpas build-in plot function currently fails
    cv.ca.stats.plot(labels=labels,
                     numbers=True,
                     cmap='gist_heat_r')
    plt.savefig(results_dir + 'confusion_matrix.png')
    if niceplot:
        ACC = cv.ca.stats.stats['mean(ACC)']
        plot_confusion(cv,
                       labels,
                       fn=results_dir + 'confusion_matrix_avmovie.svg',
                       figsize=(9, 9),
                       vmax=100,
                       cmap='Blues',
                       ACC='%.2f' % ACC)
    mv.h5save(results_dir + 'SGD_cv_classification_results.hdf5', results)
    print('Saved the crossvalidation results.')
    if store_sens:
        mv.h5save(results_dir + 'sensitivities_nfold.hdf5', sensitivities)
        print('Saved the sensitivities.')
    # results now has the overall accuracy. results.samples gives the
    # accuracy per participant.
    # sensitivities contains a dataset for each participant with the
    # sensitivities as samples and class-pairings as attributes
    return sensitivities, cv



def dotheglm(sensitivities,
             eventdir,
             annot_dir):
    """dotheglm does the glm. It will squish the sensitivity
    dataset by vstacking them, calculating the mean sensitivity per ROI pair
    with the mean_group_sample() function, transpose it with a
    TransposeMapper(). It will get the event files and read them into an apprpriate.
    data structure. It will compute one glm per run.
    """
    # normalize the sensitivities
    from sklearn.preprocessing import normalize
    import copy
    #default for normalization is the L2 norm
    sensitivities_to_normalize = copy.deepcopy(sensitivities)
    for i in range(len(sensitivities)):
         sensitivities_to_normalize[i].samples = normalize(sensitivities_to_normalize[i].samples, axis = 1)

    sensitivities_stacked = mv.vstack(sensitivities_to_normalize)
    if bilateral:
        sensitivities_stacked.sa['bilat_ROIs_str'] = map(lambda p: '_'.join(p),
                                                         sensitivities_stacked.sa.targets)
        mean_sens = mv.mean_group_sample(['bilat_ROIs_str'])(sensitivities_stacked)
    else:
        sensitivities_stacked.sa['all_ROIs_str'] = map(lambda p: '_'.join(p),
                                                         sensitivities_stacked.sa.targets)
        mean_sens = mv.mean_group_sample(['all_ROIs_str'])(sensitivities_stacked)
    mean_sens_transposed = mean_sens.get_mapped(mv.TransposeMapper())

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
    for set in setting:
        for i in range(location_annotation.setting[location_annotation['setting'] == set].value_counts()[0]):
            onset.append(location_annotation[location_annotation['setting'] == set]['onset'].values[i])
            duration.append(location_annotation[location_annotation['setting'] == set]['duration'].values[i])
        condition.append([set] * (i + 1))
    # flatten conditions
    condition = [y for x in condition for y in x]
    assert len(condition) == len(onset) == len(duration)

    # concatenate the strings
    condition_str = [set.replace(' ', '_') for set in condition]
    condition_str = ['location_' + set for set in condition_str]

    # put it in a dataframe
    locations = pd.DataFrame({
        'onset': onset,
        'duration': duration,
        'condition': condition_str
    })

    # sort according to onsets to be paranoid
    locations_sorted = locations.sort_values(by='onset')

    # this is a dataframe encoding flow of time
    time_forward = pd.DataFrame([{
        'condition': 'time+',
        'onset': location_annotation['onset'][i],
        'duration': 1.0}
        for i in range(len(location_annotation) - 1)
        if location_annotation['flow_of_time'][i] in ['+', '++']])

    time_back = pd.DataFrame([{
        'condition': 'time-',
        'onset': location_annotation['onset'][i],
        'duration': 1.0} for i in range(len(location_annotation) - 1)
        if location_annotation['flow_of_time'][i] in ['-', '--']])

    # sort according to onsets to be paranoid
    time_forward_sorted = time_forward.sort_values(by='onset')
    time_back_sorted = time_back.sort_values(by='onset')

    scene_change = pd.DataFrame([{
        'condition': 'scene-change',
        'onset': location_annotation['onset'][i],
        'duration': 1.0}
        for i in range(len(location_annotation) - 1)])

    scene_change_sorted = scene_change.sort_values(by='onset')

    # this is a dataframe encoding exterior
    exterior = pd.DataFrame([{
        'condition': 'exterior',
        'onset': location_annotation['onset'][i],
        'duration': location_annotation['duration'][i]}
        for i in range(len(location_annotation) - 1)
        if (location_annotation['int_or_ext'][i] == 'ext')])

    # sort according to onsets to be paranoid
    exterior_sorted = exterior.sort_values(by='onset')

    # this is a dataframe encoding nighttime
    night = pd.DataFrame([{'condition': 'night',
                           'onset': location_annotation['onset'][i],
                           'duration': location_annotation['duration'][i]}
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

    # check whether chunks are increasing as well as sanity check
    chunks = mean_sens_transposed.sa.chunks
    assert np.all(chunks[1:] >= chunks[:-1])

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

    # initialize the list of dicts that gets later passed to the glm
    events_dicts = []
    # This is relevant to later stack all dataframes together
    # and paranoidly make sure that they have the same columns
    cols = ['onset', 'duration', 'condition']

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
                    'condition': row['condition']
                }
                events_dicts.append(dic)

    # concatenate all event dataframes
    run_reg = pd.DataFrame([{
        'onset': runonsets[i],
        'duration': abs(runonsets[i] - runonsets[i + 1]),
        'condition': 'run-' + str(i + 1)}
        for i in range(7)])

    # get all of these wonderful dataframes into a list and squish them
    dfs = [locations_sorted[cols], scene_change_sorted[cols],
           time_back_sorted[cols], time_forward_sorted,
           exterior_sorted[cols], night_sorted[cols], run_reg[cols]]
    allevents = pd.concat(dfs)

    # save all non-face related events in an event file, just for the sake of it
    allevents.to_csv(results_dir + '/' + 'non_face_regs.tsv', sep='\t', index=False)

    # append non-faceevents to event structure for glm
    for index, row in allevents.iterrows():
        dic = {
            'onset': row['onset'],
            'duration': row['duration'],
            'condition': row['condition']
        }
        events_dicts.append(dic)

    # save this event dicts structure  as a tsv file
    import csv
    with open(results_dir + '/' + 'full_event_file.tsv', 'w') as tsvfile:
        fieldnames = ['onset', 'duration', 'condition']
        writer = csv.DictWriter(tsvfile, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        writer.writerows(events_dicts)
    # save this event file also as json file... can there ever be enough different files...
    import json
    with open(results_dir + '/' + 'allevents.json', 'w') as f:
        json.dump(events_dicts, f)

    # do the glm - we've earned it
    hrf_estimates = mv.fit_event_hrf_model(mean_sens_transposed,
                                           events_dicts,
                                           time_attr='time_coords',
                                           condition_attr='condition',
                                           design_kwargs=dict(drift_model='blank'),
                                           glmfit_kwargs=dict(model='ols'),
                                           return_model=True)

    mv.h5save(results_dir + '/' + 'sens_glm_avmovie_results.hdf5', hrf_estimates)
    print('calculated the, saving results.')

    return hrf_estimates



def makeaplot(events,
              sensitivities,
              hrf_estimates,
              roi_pair,
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

    # normalize the sensitivities
    from sklearn.preprocessing import normalize
    import copy
    #default for normalization is the L2 norm
    sensitivities_to_normalize = copy.deepcopy(sensitivities)
    for i in range(len(sensitivities)):
         sensitivities_to_normalize[i].samples = normalize(sensitivities_to_normalize[i].samples, axis = 1)

    sensitivities_stacked = mv.vstack(sensitivities_to_normalize)

    # get the mean, because we don't want to have 15 folds of sensitivities, but their average
    if bilateral:
        sensitivities_stacked.sa['bilat_ROIs_str'] = map(lambda p: '_'.join(p),
                                                         sensitivities_stacked.sa.targets)
        mean_sens = mv.mean_group_sample(['bilat_ROIs_str'])(sensitivities_stacked)
    else:
        sensitivities_stacked.sa['all_ROIs_str'] = map(lambda p: '_'.join(p),
                                                                sensitivities_stacked.sa.targets)
        mean_sens = mv.mean_group_sample(['all_ROIs_str'])(sensitivities_stacked)

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

    for j in range(len(hrf_estimates.fa.bilat_ROIs_str)):
        comparison = hrf_estimates.fa.targets[j][0]
        if (roi_pair[0] in comparison) and (roi_pair[1] in comparison):
            roi_pair_idx = j
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
            # color = colors[0]
            print(stimulus)
            condition_event_mask = events['condition'] == stimulus
            onsets = events[condition_event_mask]['onset'].values
            onsets_run = [time for time in onsets if np.logical_and(time > run_onset * 2, time < run_offset * 2)]
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
                # if there are no onsets for a particular regressor, but we want to print all
                # regressors, set i manually to 0
                rectangle = plt.Rectangle((0, 0),
                                          0,
                                          0,
                                          fc=color,
                                          alpha=0.5,
                                          label='_' * 0 + stimulus.replace(" ", "") + '(' + str(
                                              '%.2f' % beta) + ')')
                plt.gca().add_patch(rectangle)

            for i, x in enumerate(onsets_run):
                # We need the i to trick the labeling. It will attempt to plot every single occurance
                # of a stimulus with numbered labels. However, appending a '_' to the label makes
                # matplotlib disregard it. If we attach an '_' * i to the label, all but the first onset
                # get a '_' prefix and are ignored.
                r_width = durations_run[i]
                rectangle = plt.Rectangle((x, y),
                                          r_width,
                                          r_height,
                                          fc=color,
                                          alpha=0.5,
                                          label='_' * i + stimulus.replace(" ", "") + '(' + str('%.2f' % beta) + ')')
                plt.gca().add_patch(rectangle)
                plt.legend(loc=1)
                # plt.axis('scaled')
                # del colors[0]

        times = roi_sens_ds.sa.time_coords[run_onset:run_offset]

        ax.plot(times, roi_sens_ds.samples[run_onset:run_offset], '-', color='black', lw=1.0)
        # plot glm model results
        glm_model = hrf_estimates.a.model.results_[0.0].predicted[run_onset:int(run_offset), roi_pair_idx]
        # ax2 = ax.twinx()
        ax.plot(times, glm_model, '-', color='#7b241c', lw=1.0)
        model_fit = hrf_estimates.a.model.results_[0.0].R2[roi_pair_idx]
        plt.title('R squared: %.2f' % model_fit)
        if fn:
            plt.savefig(results_dir + 'timecourse_avmovie_glm_sens_{}_vs_{}_run-{}.svg'.format(roi_pair[0], roi_pair[1], run + 1))



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--inputfile', help="An hdf5 file of the avmovie "
                        "data with functional ROI information, transposed",
                        required=True)
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
                        "with the --coords flag)", default=True, type=str)
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

    args = parser.parse_args()

    # get the data
    ds_file = args.inputfile
    ds = mv.h5load(ds_file)

    # are there glm inputs?
    if args.eventdir:
        eventdir = args.eventdir
    if args.annotation:
        annot_dir = args.annotation

    results_dir = '/' + args.output + '/'
    # create the output dir if it doesn't exist
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    ds_type = args.dataset
    glm = args.glm
    bilateral = args.bilateral
    niceplot = args.niceplot
    plot_ts = args.plot_time_series
    incl_regs = args.include_all_regressors
    coords = args.coords

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
    else:
        if bilateral:
            roi_pair = ['FFA', 'PPA']
        else:
            roi_pair = ['right FFA', 'right PPA']

    # The default is to store the sensitivities
    store_sens = True

    # strip brain and potential overlaps from the brain
    if ds_type == 'stripped':
        ds = strip_ds(ds)

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
        # TODO: Do I at one point want to append the time_coordinates also to
        # TODO: the dataset with coordinates?
    # or append coordinates and get rid of fmri data is specified
    elif coords == 'only-coordinates':
        ds = get_voxel_coords(ds,
                              append=False,
                              zscore=False)
        # if there is no fmri data in the ds, don't attempt to
        # get sensitivities and only to a classification
        store_sens = False
        glm = False

    sensitivities, cv = dotheclassification(ds,
                                            bilateral=bilateral,
                                            store_sens=store_sens)
    if glm:
        hrf_estimates = dotheglm(sensitivities,
                                 annot_dir=annot_dir,
                                 eventdir=eventdir)

    if plot_ts:
        events = pd.read_csv(results_dir + 'full_event_file.tsv', sep='\t')
        makeaplot(events,
                  sensitivities,
                  hrf_estimates,
                  roi_pair,
                  fn=results_dir,
                  include_all_regressors=incl_regs)

