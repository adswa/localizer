#!/usr/bin/env python

import mvpa2.suite as mv
import numpy as np
from glob import glob
import pandas as pd
import os
from utils import (bilateralize,
                   get_avmovietimes,
                   get_events,
                   get_group_events,
                   get_known_labels,
                   get_roi_pair_idx,
                   get_voxel_coords,
                   norm_and_mean,
                   plot_confusion,
                   strip_ds,
                   buildremapper)

"""
One script to rule them all:
This script shall be able to handle all analyses.
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
    --reverse: Boolean, if given, we reverse - glm on data, subsequent classification
               on resulting betas
"""

def dotheclassification(ds,
                        classifier,
                        bilateral,
                        ds_type,
                        store_sens=True,
                        niceplot=True):
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

        ## TODO: also get the classifiers estimates, but without the infs ;)

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
    print('Set up the classifier {} for classification.'.format(classifier))
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
    print('Set up the crossvalidation, going on to compute the results.')
    results = cv(ds)
    # save classification results

    with open(results_dir + 'CV_results.txt', 'a') as f:
        f.write(cv.ca.stats.as_string(description=True))

    # printing of the confusion matrix
    # first, get the labels according to the size of dataset. This is in principle
    # superflous (get_desired_labels() would exclude brain if it wasn't in the data),
    # but it'll make sure that a permitted ds_type was specified.
    print('Plotting the confusion matrix')
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
            if 'left FEF' in ds.sa.all_ROIs:
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
            if 'left FEF' in ds.sa.all_ROIs:
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
        # get a balanced accuracy estimation bc of unbalanced class frequencies
        TPR = np.mean(cv.ca.stats.stats['TPR'])
        PPV = np.mean(cv.ca.stats.stats['PPV'])
        plot_confusion(cv,
                       labels,
                       fn=results_dir + 'CV_confusion_matrix.png',
                       figsize=(9, 9),
                       vmax=100,
                       cmap='Blues',
                       ACC='%.2f' % ACC,
                       TPR='%.2f' %TPR,
                       PPV='%.2f' %PPV)
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


def dotheglm(sensitivities,
             eventdir,
             results_dir,
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

    runs, chunks, runonsets = False, False, False
    # if we're analyzing the avmovie data, we do need the parameters above:
    if analysis == 'avmovie':
        # append proper time coordinates to the sensitivities
        mean_sens_transposed, chunks, runs, runonsets = get_avmovietimes(mean_sens_transposed)

    # get an event dict
    events_dicts = get_events(analysis=analysis,
                              eventdir=eventdir,
                              results_dir = results_dir,
                              annot_dir = annot_dir,
                              multimatch = multimatch,
                              runs = runs,
                              chunks = chunks,
                              runonsets = runonsets)

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
    # sort the roi pair in alphabetical order. This is how it is done internally to
    # assign sensitivities, and we want to match this in labeling.
    sorted_rois = sorted([roi_pair[0], roi_pair[1]])
    for run in runs:
        fig, ax = plt.subplots(1, 1, figsize=[18, 10])
        colors = ['#7b241c', '#e74c3c', '#154360', '#3498db', '#145a32', '#27ae60',
                  '#9a7d0a', '#f4d03f', '#5b2c6f', '#a569bd', '#616a6b', '#ccd1d1']
        # reverse order of ROIs in title, as weight computation for gnb is done with
        # sorted_roi[1] - sorted_roi[0]
        plt.suptitle('Timecourse of sensitivities, {} versus {}, run {}'.format(sorted_rois[1],
                                                                                sorted_rois[0],
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
                        'timecourse_localizer_glm_sens_{}_vs_{}_run-{}.svg'.format(sorted_rois[0],
                                                                                   sorted_rois[1],
                                                                                   run + 1))
    return


def makeaplot_avmovie(events,
                      sensitivities,
                      hrf_estimates,
                      roi_pair,
                      normalize,
                      bilateral,
                      classifier,
                      fn=None,
                      include_all_regressors=False,
                      multimatch_only=False):
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
    # sort the roi pair in alphabetical order. This is how it is done internally to
    # assign sensitivities, and we want to match this in labeling.
    sorted_rois = sorted([roi_pair[0], roi_pair[1]])
    for run in runs:
        fig, ax = plt.subplots(1, 1, figsize=[18, 10])
        colors = ['#7b241c', '#e74c3c', '#154360', '#3498db', '#145a32', '#27ae60',
                  '#9a7d0a', '#f4d03f', '#5b2c6f', '#a569bd', '#616a6b', '#ccd1d1']

        # reverse order of ROIs in title, as weight computation for gnb is done with
        # sorted_roi[1] - sorted_roi[0]
        plt.suptitle('Timecourse of sensitivities, {} versus {}, run {}'.format(sorted_rois[1],
                                                                                sorted_rois[0],
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

        #TMP: For the FEF analysis of my Masters Thesis I only want to plot
        #multimatch results
        if multimatch_only:
            relevant_stims = ['position_sim', 'duration_sim']
        else:
            relevant_stims = block_design

        # for each stimulus, plot a color band on top of the plot
        for stimulus in relevant_stims:
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
            elif stimulus == 'duration_sim':
                color = 'forestgreen'
                y -= 6 * r_height
            elif stimulus == 'position_sim':
                color = 'orangered'
                y -= 7 * r_height
            # get the beta corresponding to the stimulus to later use in label
            beta = roi_betas_ds.samples[hrf_estimates.sa.condition == stimulus, 0]

            if include_all_regressors and onsets_run == []:
                # if there are no onsets for a particular regressor,
                # but we want to print all
                # regressors, set i manually to 0
                rectangle = plt.Rectangle((0, 0),
                                          0,
                                          0,
                                          edgecolor=color,
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
                        'timecourse_avmovie_glm_sens_{}_vs_{}_run-{}.svg'.format(sorted_rois[0],
                                                                                 sorted_rois[1],
                                                                                 run + 1))
    return


def reverse_analysis(ds,
                     classifier,
                     bilateral,
                     results_dir,
                     ds_type,
                     eventdir,
                     annot_dir,
                     analysis,
                     store_sens=False):
    """
    This reverses the analysis. We first do a glm on the data, and subsequently do a classification
    on the resulting beta coefficients.
    ds: dataset (the transposed group dataset)
    events_dicts: dictionary of events

    """
    # step 0: transpose the data (i.e. now its non-transposed) because
    # fit_event_hrf_model needs a non-transposed dataset
    ds_transposed = ds.get_mapped(mv.TransposeMapper())
    assert ds_transposed.shape[0] < ds_transposed.shape[1]

    # get the appropriate event file
    ### Q: how to I get the events_dicts from the movie data without precomputed sensitivities for time_coords?
    # extract runs, chunks, timecoords from transposed dataset... I guess getavmovie_time.. does this for me
    chunks, runs, runonsets = False, False, False

    if analysis == 'avmovie':
        ds_transposed, chunks, runs, runonsets = get_avmovietimes(ds_transposed)

    events_dicts = get_events(analysis=analysis,
                              eventdir=eventdir,
                              results_dir=results_dir,
                              chunks=chunks,
                              runs=runs,
                              runonsets=runonsets,
                              annot_dir=annot_dir,
                              multimatch=multimatch)

    # step 1: do the glm on the data
    hrf_estimates = mv.fit_event_hrf_model(ds_transposed,
                                           events_dicts,
                                           time_attr='time_coords',
                                           condition_attr='condition',
                                           design_kwargs=dict(drift_model='blank'),
                                           glmfit_kwargs=dict(model='ols'),
                                           return_model=True)
    # now what is this? I will try to figure it out from a stripped localizer dataset. We get
    # hrf_estimates_transposed.samples.shape --> (16985, 12) (12 regressors, 16985 voxels). Question: In dotheglm,
    # we provide mean sensitivities. here, ds should still contain all 15 subjects data (16985 voxel seem to be all
    # combined voxels for all participants)... We need to find a way to get betas per participant and preserve
    # participant information in the resulting dataset. - WAIT. This appears to be already the case.
    # What we would need is a plot of the glm. what would we plot there... currently its voxels x regressors instead
    # of time x regressors... we could do voxels grouped into ROIs on x axis
    # .


    # lets save these
    mv.h5save(results_dir + '/' + 'sens_glm_results.hdf5', hrf_estimates)
    print('calculated the glm, saving results')

    # step 2: get the results back into a transposed form, because we want
    # to have time points as features & extract the betas
    hrf_estimates_transposed = hrf_estimates.get_mapped(mv.TransposeMapper())
    assert hrf_estimates_transposed.samples.shape[0] > hrf_estimates_transposed.samples.shape[1]

    # find out what is happening in a given ROI. For this, use the transposed hrf_estimates,
    # and project them into a brain.
    print('going on to project resulting betas back into brain...')
    subs = np.unique(hrf_estimates_transposed.sa.participant)
    regs = hrf_estimates_transposed.fa.condition
    assert len(subs) > 0
    from collections import OrderedDict
    result_maps = OrderedDict()
    # for sub in subs:
    # for now one sub
    for sub in subs:
        print('...for subject {}...'.format(sub))
        result_maps[sub] = OrderedDict()
        # subset to participants dataframe
        data = mv.Dataset(hrf_estimates_transposed.samples[hrf_estimates_transposed.sa.participant == sub],
                          fa=hrf_estimates_transposed[hrf_estimates_transposed.sa.participant == sub].fa,
                          sa=hrf_estimates_transposed[hrf_estimates_transposed.sa.participant == sub].sa)
        # loop over regressors
        for idx, reg in enumerate(regs):
            result_map = buildremapper(ds_type,
                                       sub,
                                       data.samples.T[idx], # we select one beta vector per regressor
                                       )
            # populate a nested dict with the resulting nifti images
            # this guy has one nifti image per regressor for each subject
            result_maps[sub][reg] = result_map

        # Those result maps can be quick-and-dirty-plotted with
        # mri_args = {'background' : 'sourcedata/tnt/sub-01/bold3Tp2/in_grpbold3Tp2/head.nii.gz',
        # 'background_mask': 'sub-01/ses-movie/anat/brain_mask_tmpl.nii.gz'}
        # fig = mv.plot_lightbox(overlay=result_maps['sub-01']['scene'], vlim=(1.5, None), **mri_args)
        # TODO: maybe save the result map? Done with map2nifti(ds, da).to_filename('blabla{}'.format(reg)
        # how do we know which regressors have highest betas for given ROI? averaging?
        # hrf_estimates.samples[0][hrf_estimates.fa.bilat_ROIs == 'FFA'] gives a beta vector for one reg
        # [np.mean(hrf_estimates.samples[i][hrf_estimates.fa.bilat_ROIs == 'PPA']) for i, reg in enumerate(regs)]
        # FFA_body: 1.146           PPA: 0.191
        # FFA body first: 2.929     PPA: 0.955
        # FFA face: 1.573           PPA: 0.089
        # FFA face first: 3.191     PPA: 0.025
        # FFA house: 0.638          PPA: 1.06
        # FFA house first: 2.270    PPA: 2.60
        # FFA object: 0.735         PPA: 0.32
        # FFA object first: 2.17    PPA: 1.27
        # FFA scene: 0.405          PPA: 1.26
        # FFA scene first: 1.405    PPA; 3.959
        # FFA scramnle: 0.285       PPA: 0.31
        # FFA scramble first: 0.521 PPA: 1.28
        #from collections import OrderedDict
        #betas = [np.mean(hrf_estimates.samples[i][hrf_estimates.fa.bilat_ROIs == 'PPA']) for i, reg in enumerate(regs)]
        # to get it sorted: OrderedDict(sorted(zip(regs, betas), key=lambda x:x[1]))

    # step 3: do the classification on the betas. We do not store sensitivies (as no glm is necessary
    # anymore)

    # currently, I get one beta per regressor per voxel for the movie data.
    # The classification accuracy is poor for the movie data set when using all regressors (.24).
    # If I include only the face regressors it increases substantially (0.46 with an l-sgd for the stripped dataset)
    sensitivities, cv = dotheclassification(hrf_estimates_transposed,
                                            classifier,
                                            bilateral,
                                            ds_type,
                                            store_sens=False,
                                            niceplot = niceplot)

    # This prints a confusion matrix. the question is, how to we interprete these results?
    return hrf_estimates_transposed, sensitivities, cv


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-i', '--inputfile',
        help="An hdf5 file of the avmovie data with functional ROI information, transposed",
        required=True
    )
    parser.add_argument(
        '--analysis',
        help="[If glm is computed:] Which dataset is the analysis based on, 'localizer' or 'avmovie'",
        type=str
    )
    parser.add_argument(
        '-a', '--annotation',
        help="Input a single, full movie spanning location annotation file, if you want to compute the glm."
    )
    parser.add_argument(
        '-e', '--eventdir',
        help="Input the directory name under which the downsamples, run-wise event files can be found, if you want "
             "to compute the glm. (localizer e.g. 'sourcedata/phase2/*/ses-localizer/func/' movie e.g. "
             "'derivatives/stimuli/researchcut/downsampled_event_files/'"
    )
    parser.add_argument(
        '-bi', '--bilateral',
        help="If false, computation will be made on hemisphere-specific ROIs (i.e. left FFA, right FFA",
        action='store_true'
    )
    parser.add_argument(
        '-g', '--glm',
        help="Should a glm on the sensitivities be computed? Defaults to True, as long as the classification isn't "
             "done on an only-coordinates dataset (as specified with the --coords flag)",
        action='store_true'
    )
    parser.add_argument(
        '-ds', '--dataset',
        help="Specify whether the analysis should be done on the full dataset or on the dataset with only ROIs: "
             "'full' or 'stripped' (default: stripped)",
        type=str,
        default='stripped'
    )
    parser.add_argument(
        '-c', '--coords',
        help="Should coordinates be included in the dataset? ('with-coordinates').Should a sanity check with only "
             "coordinates without fmri data be performed? ('only-coordinates'). Should coordinates be disregard? ("
             "'no-coordinates') Default: 'no-coordinates'.",
        type=str,
        default='no-coordinates'
    )
    parser.add_argument(
        '-o', '--output',
        help="Please specify an output directory name (absolute path) to store the analysis results",
        type=str
    )
    parser.add_argument(
        '-r', '--roipair',
        nargs='+',
        help="Specify two ROIs for which the glm timecourse should be plotted. Default for now is right FFA & right "
             "PPA in lateralized dataset, FFA & PPA in bilateral dataset. Specify as --roipair 'FFA' 'PPA'"
    )
    parser.add_argument(
        '-n', '--niceplot',
        help="If true, the confusion matrix of the classification will be plotted with Matplotlib instead of build "
             "in functions of pymvpa."
    )
    parser.add_argument(
        '-ps', '--plot_time_series',
        help="If True, the results of the glm will be plotted as a timeseries per run.",
        action='store_true'
    )
    parser.add_argument(
        '-ar', '--include_all_regressors',
        help="If you are plotting the time series, do you want the plot to contain all of the regressors?",
        action='store_true'
    )
    parser.add_argument(
        '--classifier',
        help="Which classifier do you want to use? Options: linear Gaussian Naive Bayes ('gnb'), linear (binary) "
             "stochastic gradient descent (l-sgd), stochastic gradient descent (sgd)",
        type=str,
        required=True
    )
    parser.add_argument(
        '--normalize',
        help="Should the sensitivities used for the glm be normalized by their L2 norm? True/False",
        action='store_true'
    )
    parser.add_argument(
        '--multimatch',
        help="path to multimatch mean results per run. If given, the similarity measures for position and duration "
             "will be included in the avmovie glm analysis. Provide path including file name, "
             "as in 'sourcedata/multimatch/output/run_*/means.tsv'"
    )
    parser.add_argument(
        '--multimatch-only',
        help='TMPargs, if I only want to plot multimatch regressors',
        action='store_true'
    )
    parser.add_argument(
        '--reverse',
        help='If given, the analysis is reversed (first glm on data, subsequent classification on betas)',
        action='store_true'
    )

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
    glm = True if args.glm else False           # True or False
    bilateral = True if args.bilateral else False # True or False
    niceplot = True if args.niceplot else False   # False or True
    normalize = True if args.normalize else False # True or False
    plot_ts = True if args.plot_time_series else False  # False or True
    incl_regs = args.include_all_regressors
    coords = args.coords                        # no-coords --> leave ds as is,
                                                # with-coords --> incl. coords,
                                                # only-coords --> only coords
    classifier = args.classifier                # gnb, sgd, l-sgd --> multiclassclassifier
    annot_dir = args.annotation if args.annotation else None
    multimatch = False
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
    if glm or args.reverse:
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

            multimatch_only = args.multimatch_only
            if multimatch_only:
                print('I will plot only multimatch regressors')

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
    else:
        if bilateral:
            roi_pair = ['FFA', 'PPA']
        else:
            roi_pair = ['right FFA', 'right PPA']

    # if the ROI 'brain' is specified in the comparison, rename everything that is not
    # the other ROI in question to really have a ROI-vs-brain distinction
    if 'brain' in roi_pair:
        keep_roi = [roi for roi in roi_pair if roi != 'brain']
        if ds_type == 'stripped':
            raise ValueError(
                """You specified to compute a roi pair that contains the roi 'brain',
                but you also stripped the dataset so that there is no 'brain' anymore.
                Computer says no."""
            )
        if bilateral:
            ds.sa.bilat_ROIs[ds.sa.bilat_ROIs != keep_roi[0]] = 'brain'
        else:
            ds.sa.all_ROIs[ds.sa.all_ROIs != keep_roi[0]] = 'brain'
        print('Relabeled everything but {} to "brain".'.format(keep_roi))
    print("If we're doing a GLM, this ROI pair is going to be used: {}".format(roi_pair))

        ## TODO: what happens to roi pair in the event of a sgd classifier 1-vs-all?
    # currently, related to the todo, the glm computation won't work in sgd
    # 1-vs-all classification as we can't derive the comparison roi yet. For now,
    # I will disable glm computation for these cases.
    if classifier == 'sgd':
        glm = False
        print("Currently, the glm computation won't work in sgd 1-vs-all classification as we can't derive the "
              "comparison roi yet. For now,the glm computation for these cases is disabled")

    if args.reverse:
        # swap sequence of computations: first glm on data, then classification based on betas
        hrf_estimates_transposed, sensitivities, cv = reverse_analysis(ds,
                                                                       classifier,
                                                                       bilateral,
                                                                       results_dir,
                                                                       ds_type,
                                                                       eventdir,
                                                                       annot_dir,
                                                                       analysis,
                                                                       store_sens=False)
    else:
        # do the "normal" sequence: first classification, then GLM on derived sensitivities
        sensitivities, cv = dotheclassification(ds,
                                                classifier=classifier,
                                                bilateral=bilateral,
                                                ds_type=ds_type,
                                                store_sens=store_sens,
                                                niceplot=niceplot)
        if (glm) and (analysis == 'avmovie'):
            hrf_estimates = dotheglm(sensitivities,
                                     normalize=normalize,
                                     classifier=classifier,
                                     analysis=analysis,
                                     annot_dir=annot_dir,
                                     eventdir=eventdir,
                                     results_dir=results_dir,
                                     multimatch=multimatch
                                     )
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
                                  include_all_regressors=incl_regs,
                                  multimatch_only=multimatch_only)
        elif (glm) and (analysis == 'localizer'):
            hrf_estimates = dotheglm(sensitivities,
                                     normalize=normalize,
                                     analysis=analysis,
                                     classifier=classifier,
                                     eventdir=eventdir,
                                     results_dir = results_dir,
                                     multimatch=False)
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


