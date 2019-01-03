#!/home/adina/wtf/bin/python

import numpy as np
import mvpa2.suite as mv
from glob import glob
import pandas as pd
import itertools
import os

"""
This script takes an existing transposed hdf5 dataset, containing the studyforrest
objectcategories localizer paradigm data and the data of the functional ROIs, as its input file.

To call the dataset, specify the following command line options/inputs:
BLABLABLA

This script uses a support vector machine as classifying algorithm of choice.
TODO: I might want to integrate a classifier choice into a general analysis file.
For now, it lives in separate scripts, but addind the SGD to the already existing
file with GLM might be worth a thought in the future.
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
    """ Start of a helper function to reorder ROI labels in a confusion matrix."""
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


def get_group_events(eventdir):
    event_files = sorted(glob(eventdir + '*_events.tsv'))
    if len(event_files) == 0:
        print('No event files were discovered. Make sure you only specify the directory'
              'the eventfiles are in, and not the names of the eventfiles. The way the'
              'event files are globbed is glob(eventdir + "*_events.tsv").')
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

    # returns and event file array
    return events


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
    im = sns.heatmap(100*matrix.astype(float)/np.sum(matrix, axis=1)[:, None],
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
#    import matplotlib.pyplot as plt
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

    with open(results_dir + 'objectcategories_clf.txt', 'a') as f:
        f.write(cv.ca.stats.as_string(description=True))
    # printing of the confusion matrix
    if bilateral:
        desired_order = ['VIS', 'LOC', 'OFA', 'FFA', 'EBA', 'PPA']
    else:
        desired_order = ['brain', 'VIS', 'left LOC', 'right LOC',
                         'left OFA', 'right OFA', 'left FFA',
                         'right FFA', 'left EBA', 'right EBA',
                         'left PPA', 'right PPA']

    labels = get_known_labels(desired_order, cv.ca.stats.labels)

    # plot the confusion matrix with pymvpas build-in plot function currently fails
#    cv.ca.stats.plot(labels=labels,
#                     numbers=True,
#                     cmap='gist_heat_r')
#    plt.savefig(results_dir + 'confusion_matrix.png')
#    if niceplot:
#        ACC = cv.ca.stats.stats['mean(ACC)']
#        plot_confusion(cv,
#                       labels,
#                       fn=results_dir + 'confusion_matrix_avmovie.svg',
#                       figsize=(9, 9),
#                       vmax=100,
#                       cmap='Blues',
#                       ACC='%.2f' % ACC)
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


def dotheglm(sensitivities, eventdir):
    """dotheglm does the glm. It will squish the sensitivity
    dataset by vstacking them, calculating the mean sensitivity per ROI pair
    with the mean_group_sample() function, transpose it with a
    TransposeMapper(). It will get the event files and read them in, average the
    durations because there are tiny differences between subjects, and then it
    will put all of that into a glm.
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
            'condition': events[i][2]
        }
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


def makeaplot(events,
              sensitivities,
              hrf_estimates,
              roi_pair,
              fn=True):
    """
    This produces a time series plot for the roi class comparison specified in
    roi_pair such as roi_pair = ['left FFA', 'left PPA']
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

    if bilateral:
        sensitivities_stacked.sa['bilat_ROIs_str'] = map(lambda p: '_'.join(p),
                                                         sensitivities_stacked.sa.targets)
        mean_sens = mv.mean_group_sample(['bilat_ROIs_str'])(sensitivities_stacked)
    else:
        sensitivities_stacked.sa['all_ROIs_str'] = map(lambda p: '_'.join(p),
                                                                sensitivities_stacked.sa.targets)
        mean_sens = mv.mean_group_sample(['all_ROIs_str'])(sensitivities_stacked)

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

    for j in range(len(hrf_estimates.fa.bilat_ROIs_str)):
        comparison = hrf_estimates.fa.targets[j][0]
        if (roi_pair[0] in comparison) and (roi_pair[1] in comparison):
            roi_pair_idx = j
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
                                          label='_'*i + stimulus.replace(" ", "") + '(' + str('%.2f' % beta) + ')')
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
            plt.savefig(results_dir + 'timecourse_localizer_glm_sens_{}_vs_{}_run-{}.svg'.format(roi_pair[0], roi_pair[1], run + 1))




if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--inputfile', help="An hdf5 file of the localizer"
                        "data with functional ROI information, transposed",
                        required=True)
    parser.add_argument('-e', '--eventdir', help="Input the directory name under which"
                        "the eventfiles of the block design can be found for the "
                        "participants, if you want to compute the glm. If they are"
                        "under subject specific subdirectory, specify a path with"
                        "appropriate wildcards.")
    parser.add_argument('-bi', '--bilateral', help="If false, computation will "
                        "be made on hemisphere-specific ROIs (i.e. left FFA, "
                        "right FFA", default=True)
    parser.add_argument('-g', '--glm', help="Should a glm on the sensitivities be"
                        "computed? Defaults to True, as long as the classification"
                        "isn't done on an only-coordinates dataset (as specified "
                        "with the --coords flag)", default=True, type=str)
    parser.add_argument('-ds', '--dataset', help="Specify whether the analysis"
                        "should be done on the full dataset or on the dataset"
                        "with only ROIs: 'full' or 'stripped' (default: stripped)",
                        type=str, default='stripped')
    parser.add_argument('-c', '--coords', help="Should coordinates be included in"
                        "the dataset? ('with-coordinates').Should a sanity check"
                        "with only coordinates without fmri data be performed?"
                        "('only-coordinates'). Should coordinates be disregarded?"
                        "('no-coordinates') Default: 'no-coordinates'.", type=str,
                        default='no-coordinates')
    parser.add_argument('-o', '--output', help="Please specify an output directory"
                        "name (absolute path) to store the analysis results", type=str)
    parser.add_argument('-r', '--roipair', nargs='+', help="Specify two ROIs for which the glm"
                        "timecourse should be plotted. Default for now is right FFA &"
                        "right PPA in lateralized dataset, FFA & PPA in bilateral"
                        "dataset. Specify as --roipair 'FFA' 'PPA'")
    parser.add_argument('-n', '--niceplot', help="If true, the confusion matrix of the"
                        " classification will be plotted with Matplotlib instead of"
                        " build in functions of pymvpa. ON HYDRA THIS WILL CRASH!",
                        default=False)
    parser.add_argument('-ps', '--plot_time_series', help="If True, the results of"
                        " the glm will be plotted as a timeseries per run.", default=False)
    parser.add_argument('-ar', '--include_all_regressors', help="If you are plotting the time series, do you want"
                        "the plot to contain all of the regressors?", default=False)
    args = parser.parse_args()

    # get the data
    ds_file = args.inputfile
    ds = mv.h5load(ds_file)

    # are there glm inputs?
    if args.eventdir:
        eventdir = '/' + args.eventdir + '/'

    results_dir = '/' + args.output + '/'
    # create the output dir if it doesn't exist
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    ds_type = args.dataset
    glm = args.glm
    coords = args.coords
    bilateral = args.bilateral
    niceplot = args.niceplot
    plot_ts = args.plot_time_series
    incl_regs = args.include_all_regressors
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
        #  get sensitivities and only to a classification
        store_sens = False
        glm = False

    sensitivities, cv = dotheclassification(ds,
                                            bilateral=bilateral,
                                            store_sens=store_sens)

    if glm:
        hrf_estimates = dotheglm(sensitivities,
                                eventdir)

    if plot_ts:
        # read the event files, they've been produced by the glm
        events = pd.read_csv(results_dir + 'group_events.tsv',
                             sep='\t')
        makeaplot(events,
                  sensitivities,
                  hrf_estimates,
                  roi_pair,
                  fn=results_dir)

