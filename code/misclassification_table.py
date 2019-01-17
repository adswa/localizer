#!/usr/bin/env python

import mvpa2.suite as mv
import numpy as np
import pandas as pd
import os


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
    # mv.h5save(results_dir + 'ds_ROIs.hdf5', ds_ROIs)
    print('Combined lateralized ROIs for the provided dataset.')
    return ds_ROIs


def dotheclassification(ds_movie,
                        ds_loc,
                        classifier,
                        bilateral):
    """ Dotheclassification does the classification.
    Input: the dataset on which to perform a leave-one-out crossvalidation with a classifier
    of choice.
    Specify: the classifier to be used (gnb (linear gnb), l-sgd (linear sgd), sgd)
             whether the sensitivities should be computed and stored for later use
             whether the dataset has ROIs combined across hemisphere (bilateral)
    """

    dfs = []
    for idx, ds in enumerate([ds_movie, ds_loc]):
        if bilateral:
            ds.sa['targets'] = ds.sa.bilat_ROIs
        else:
            ds.sa['targets'] = ds.sa.all_ROIs

        if classifier == 'gnb':
            # set up classifier
            prior = 'ratio'
            clf = mv.GNB(common_variance=True,
                         prior=prior)

        elif classifier == 'sgd':
            # necessary I believe regardless of the SKLLearnerAdapter
            from sklearn.linear_model import SGDClassifier
            clf = mv.SKLLearnerAdapter(SGDClassifier(loss='hinge',
                                                     penalty='l2',
                                                     class_weight='balanced'))
        elif classifier == 'l-sgd':
            # necessary I believe regardless of the SKLLearnerAdapter
            from sklearn.linear_model import SGDClassifier
            # get a stochastic gradient descent into pymvpa by using the SKLLearnerAdapter.
            # Get it to perform 1 vs 1 decisions (instead of one vs all) with the MulticlassClassifier
            clf = mv.MulticlassClassifier(mv.SKLLearnerAdapter(SGDClassifier(loss='hinge',
                                                                             penalty='l2',
                                                                             class_weight='balanced'
                                                                             )))

        # prepare for callback of sensitivity extraction within CrossValidation
        classifications = []

        def store_class(data, node, result):
            # import pdb; pdb.set_trace()
            class_ds = mv.Dataset(samples=data.sa.voxel_indices)
            class_ds.sa['targets'] = data.sa.targets
            class_ds.sa['partitions'] = data.sa.partitions
            class_ds.sa['predictions'] = clf.predict(data)
            class_ds.sa['participant'] = data.sa.participant
            classifications.append(class_ds)

        # do a crossvalidation classification and store the classification results
        cv = mv.CrossValidation(clf, mv.NFoldPartitioner(attr='participant'),
                                errorfx=mv.mean_match_accuracy,
                                enable_ca=['stats'],
                                callback=store_class)
        # import pdb; pdb.set_trace()
        results = cv(ds)
        # import pdb; pdb.set_trace()
        # save classification results as a Dataset
        ds_type = ['movie', 'loc']
        mv.h5save(results_dir + 'cv_classification_results_{}.hdf5'.format(ds_type[idx]), classifications)
        print('Saved the classification results obtained during crossvalidation.')

        # get the classification list into a pandas dataframe

        for i, classification in enumerate(classifications):
            df = pd.DataFrame(data={'voxel_indices': list(classification.samples),
                                    'targets': list(classification.sa.targets),
                                    'predictions': list(classification.sa.predictions),
                                    'partitions': list(classification.sa.partitions),
                                    'participants': list(classification.sa.participant),
                                    'ds_type': [ds_type[idx]] * len(classification.sa.predictions)
                                    }
                              )
            dfs.append(df)

    # two helper functions for later use in a lamda function
    def hits(row):
        if row['predictions'] == row['targets']:
            return 1
        else:
            return 0

    def parts(row):
        if row['partitions'] == 1:
            return "train"
        elif row['partitions'] == 2:
            return "test"

    # get all folds into one dataframe, disregard the index
    all_classifications = pd.concat(dfs, ignore_index=True)
    # compute hits as correspondence between target and prediction
    all_classifications['hits'] = all_classifications.apply(lambda row: hits(row), axis=1)
    # assign string labels to testing and training partitions (instead of 1, 2)
    all_classifications['parts'] = all_classifications.apply(lambda row: parts(row), axis=1)
    # transform voxel coordinates from arrays (unhashable) into tuples
    all_classifications['voxel_indices'] = all_classifications['voxel_indices'].apply(tuple)

    # subset the dataset to contain only the testing data
    all_testing = all_classifications[all_classifications.parts == "test"]
    # check that every participant is in the data
    assert len(all_testing.participants.unique()) == 15
    # to check for correspondence between the sum of the two experiments confusion matrices,
    # do sth like this: len(all_testing[(all_testing['predictions'] == 'PPA') & (all_testing['targets'] == 'VIS')])

    # this counts hits per fold across experiments (2 if both experiments classified correctly,
    # 1 if 1 experiment classified correctly, 0 is none did). Also, append the targets per voxel.
    # we use 'min' here because aggregate needs any function, but targets are the same between
    # the experiments
    compare_exp = all_testing.groupby(['voxel_indices', 'participants']).agg(
        {'hits': 'sum', 'targets': 'min'}).reset_index().sort_values(['voxel_indices', 'participants'])
    all_testing_movie = all_testing[all_testing.ds_type == 'movie'].sort_values(
        ['voxel_indices', 'participants']).reset_index()
    all_testing_loc = all_testing[all_testing.ds_type == 'loc'].sort_values(
        ['voxel_indices', 'participants']).reset_index()
    # append movie and loc predictions to the dataframe
    compare_exp['pred_movie'] = all_testing_movie.predictions
    compare_exp['pred_loc'] = all_testing_loc.predictions

    # get the ROIS from the classification
    ROIS = np.unique(ds_movie.sa.targets)

    # there can't be values greater than two or lower than zero
    assert compare_exp.hits.max() <= 2
    assert compare_exp.hits.min() >= 0
    return compare_exp, all_testing, ROIS


def dice_matrix(ROIS,
                compare_exp,
                bilateral=True,
                ds_type='stripped',
                plotting=True):
    """
    This function plots a matrix of dice coefficients between the two
    datasets classification results. The dice coefficients indicates
    how many voxels were classified identically between the datasets.
    :param ROIS: target sample attributes
    :param compare_exp: dataframe of combined classification results
    :param bilateral: True-->ROIs combined across hemispheres
    :param ds_type: 'stripped' (only rois) or 'full' (with brain)
    :param plotting: if True, a confusion matrix (heatmap) is plotted
    :return:
    """

    indices = []
    dice_coeffs = []
    for roi_row in ROIS:
        for roi_col in ROIS:
            intersection = len(compare_exp[(compare_exp.pred_movie == roi_row) &
                                           (compare_exp.pred_loc == roi_col) &
                                           (compare_exp.targets == roi_col)])
            movie_card = len(compare_exp[compare_exp.pred_movie == roi_row])
            loc_card = len(compare_exp[compare_exp.pred_loc == roi_row])
            dice = 2 * intersection / (float(movie_card) + float(loc_card))
            dice_coeffs.append(dice)
        indices.append(roi_row)
    dice_m = np.asarray(dice_coeffs).reshape((len(indices), len(indices)))
    unordered = pd.DataFrame(dice_m, columns=indices, index=indices)
    if ds_type == 'full':
        if bilateral:
            desired_order = ['brain', 'VIS', 'LOC', 'OFA', 'FFA', 'EBA', 'PPA']
        else:
            desired_order = ['brain', 'VIS', 'left LOC', 'right LOC',
                             'left OFA', 'right OFA', 'left FFA',
                             'right FFA', 'left EBA', 'right EBA',
                             'left PPA', 'right PPA']
    if ds_type == 'stripped':
        if bilateral:
            desired_order = ['VIS', 'LOC', 'OFA', 'FFA', 'EBA', 'PPA']
        else:
            desired_order = ['VIS', 'left LOC', 'right LOC',
                             'left OFA', 'right OFA', 'left FFA',
                             'right FFA', 'left EBA', 'right EBA',
                             'left PPA', 'right PPA']

    sim_matrix = unordered[desired_order].reindex(desired_order)
    if plotting:
        # plot the matrix as a heatmap
        import matplotlib.pyplot as plt
        import seaborn as sn
        plt.figure()
        sn.heatmap(sim_matrix, annot=True, cmap='Blues')
        plt.savefig(results_dir + 'heatmap_of_dice_coefficients.png')

    return sim_matrix


def calc_sim_metrics(compare_exp,
                     all_testing,
                     ROIS):
    """Calculates all kinds of similarity metrics between classifications
    between datasets and returns themm as an Ordered Dictionary. Yes, Yarik,
    its reeeeaally ugly. Its just a helper."""

    names = []
    voxelcounts = []
    common_hitss = []
    card_movies = []
    card_locs = []
    misclass_movies = []
    misclass_locs = []
    DSCs = []
    movie_alone_abss = []
    movie_alone_proportions = []
    loc_alone_abss = []
    loc_alone_proportions = []
    prop_found_by_both = []
    prop_found_by_none = []
    for ROI in ROIS:
        names.append(str(ROI))
        voxelcount = len(all_testing[all_testing.targets == ROI]) / 2
        voxelcounts.append(voxelcount)
        common_hits = len(compare_exp[(compare_exp.targets == ROI) & (compare_exp.hits == 2)])
        common_hitss.append(common_hits)
        card_movie = len(all_testing[(all_testing.ds_type == 'movie') & (all_testing.predictions == ROI)])
        card_movies.append(card_movie)
        card_loc = len(all_testing[(all_testing.ds_type == 'loc') & (all_testing.predictions == ROI)])
        card_locs.append(card_loc)
        misclass_movie = len(all_testing[(all_testing.ds_type == 'movie') &
                                         (all_testing.targets != ROI) &
                                         (all_testing.predictions == ROI)])
        misclass_movies.append(misclass_movie)
        misclass_loc = len(all_testing[(all_testing.ds_type == 'loc') &
                                       (all_testing.targets != ROI) &
                                       (all_testing.predictions == ROI)])
        misclass_locs.append(misclass_loc)
        # compute the dice index from intersection / sum cardinalities of the sets (experiments)
        # import pdb; pdb.set_trace()
        DSC = (float(2 * common_hits)) / (float(card_loc + card_movie))
        DSCs.append(DSC)
        # calculate the number and portion of correctly classified voxels NOT correctly classified
        # by the other experiment
        movie_alone_abs = card_movie - (misclass_movie + common_hits)
        movie_alone_abss.append(movie_alone_abs)
        movie_alone_proportion = float(movie_alone_abs) / float(voxelcount)
        movie_alone_proportions.append(movie_alone_proportion)
        loc_alone_abs = card_loc - (misclass_loc + common_hits)
        loc_alone_abss.append(loc_alone_abs)
        loc_alone_proportion = float(loc_alone_abs) / float(voxelcount)
        loc_alone_proportions.append(loc_alone_proportion)
        prop_found_by_both.append(float(common_hits) / float(voxelcount))
        prop_found_by_none.append(float(len(compare_exp[(compare_exp.hits == 0) &
                                                        (compare_exp.targets == ROI)]))
                                  / float(voxelcount))

    # put all of this lists into an ordered dict as well, just to have them available.
    from collections import OrderedDict
    attribs = OrderedDict()
    attribs['name'] = names
    attribs['voxelcount'] = voxelcounts
    attribs['common_hits'] = common_hitss
    attribs['prop_found_by_both'] = prop_found_by_both
    attribs['prop_found_by_none'] = prop_found_by_none
    attribs['movie_alone_props'] = movie_alone_proportions
    attribs['loc_alone_props'] = loc_alone_proportions
    attribs['card_movie'] = card_movies
    attribs['card_loc'] = card_locs
    attribs['misclass_movie'] = misclass_movies
    attribs['misclass_loc'] = misclass_locs
    attribs['DSC'] = DSCs
    attribs['movie_alone_abs'] = movie_alone_abss
    attribs['loc_alone_abs'] = loc_alone_abss

    return prop_found_by_both, movie_alone_proportions, \
           loc_alone_proportions, prop_found_by_none, attribs
    # plot as horizontal bar plot:


def plot_voxelclf_per_ds(compare_exp,
                         all_testing,
                         ROIS):
    """
    Builds and ordered dict of matrics to compare voxel classifications per ROI from,
    and plots the proportion of voxel per roi that are...
    - correctly classified in both datasets
    - only correctly classified in the movie dataset
    - only correctly classified in the localizer dataset
    - never correctly classified
    """
    # compute dice indices and for localizer and movie experiment, per ROI,
    # compute the amount of voxels only one experiment could classify correctly.
    both, movie, loc, none, attribs = calc_sim_metrics(compare_exp,
                                                       all_testing,
                                                       ROIS)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    n = list(range(len(ROIS)))
    barwidth = 0.8
    plt.bar(n,
            both,
            color='olive',
            width=barwidth)
    plt.bar(n,
            movie,
            bottom=both,
            color='yellowgreen',
            width=barwidth)
    plt.bar(n,
            loc,
            bottom=[sum(x) for x in zip(movie, both)],
            color='coral',
            width=barwidth)
    plt.bar(n,
            none,
            bottom=[sum(x) for x in zip(movie, both, loc)],
            color='cadetblue',
            width=barwidth)
    plt.xticks(n, ROIS)
    legend_elements = [Line2D([0], [0], color='olive', lw=5, label='both'),
                       Line2D([0], [0], color='yellowgreen', lw=5, label='movie'),
                       Line2D([0], [0], color='coral', lw=5, label='localizer'),
                       Line2D([0], [0], color='cadetblue', lw=5, label='none')]
    plt.legend(handles=legend_elements)
    plt.title('Comparison of individual voxel classification in both experiments ')
    plt.savefig(results_dir + 'classification_count_per_ROI.png')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--inputfile1', help="An hdf5 file of the avmovie "
                                                   "data with functional ROI information, transposed",
                        required=True)
    parser.add_argument('-j', '--inputfile2', help="An hdf5 file of the localizer "
                                                   "data with functional ROI information, transposed",
                        required=True)
    parser.add_argument('-bi', '--bilateral', help="If false, computation will "
                                                   "be made on hemisphere-specific ROIs (i.e. left FFA, "
                                                   "right FFA", default=True)
    parser.add_argument('-ds', '--dataset', help="Specify whether the analysis \
                        should be done on the full dataset or on the dataset \
                        with only ROIs: 'full' or 'stripped' (default: stripped)",
                        type=str, default='stripped')
    parser.add_argument('-o', '--output', help="Please specify an output directory"
                                               "name (absolute path) to store the analysis results", type=str)
    parser.add_argument('--classifier', help="Which classifier do you want to use? Options:"
                                             "linear Gaussian Naive Bayes ('gnb'), linear (binary) stochastic "
                                             "gradient descent (l-sgd)",
                        type=str, required=True)

    args = parser.parse_args()

    # get the data
    ds_movie_path = args.inputfile1
    ds_loc_path = args.inputfile2
    ds_movie = mv.h5load(ds_movie_path)
    ds_loc = mv.h5load(ds_loc_path)

    # prepare the output path
    results_dir = '/' + args.output + '/'
    # create the output dir if it doesn't exist
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    # get more information about what is being calculated
    ds_type = args.dataset  # stripped --> no brain, no overlap,

    if args.bilateral == 'True' or args.bilateral == True:
        bilateral = True
    elif args.bilateral == 'False' or args.bilateral == False:
        bilateral = False  # True or False

    classifier = args.classifier  # gnb, sgd, l-sgd --> multiclassclassifier

    # fail early, if classifier is not appropriately specified.
    allowed_clfs = ['l-sgd', 'sgd', 'gnb']
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

    if ds_type == 'stripped':
        ds_movie = strip_ds(ds_movie, order='full')
        ds_loc = strip_ds(ds_loc, order='full')
    else:
        ds_movie = strip_ds(ds_movie, order='sparse')
        ds_loc = strip_ds(ds_loc, order='sparse')
    # combine ROIs of the hemispheres
    if bilateral:
        ds_movie = bilateralize(ds_movie)
        ds_loc = bilateralize(ds_loc)

    compare_exp, all_testing, ROIS = dotheclassification(ds_movie,
                                                         ds_loc,
                                                         classifier,
                                                         bilateral)

    plot_voxelclf_per_ds(compare_exp,
                         all_testing,
                         ROIS)

    dice_matrix(ROIS,
                compare_exp,
                bilateral=True,
                ds_type='stripped',
                plotting=True)

