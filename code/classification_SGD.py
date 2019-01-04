#!/usr/bin/env python

import mvpa2.suite as mv
from sklearn.linear_model import SGDClassifier
import numpy as np
import os
import matplotlib.pyplot as plt

"""
This script performs a classification with stochastic gradient descent on
1-vs-everything else in a leave-one-out crossvalidation on the full brain data.
It largely copies Sams SGD_localizer script (thanks).
"""


def strip_ds(ds):
    """Helper to get rid of only overlap in the dataset."""
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


def dotheclassification(ds,
                        bilateral):
    """This functions performs the classification in a one-vs-all fashion with a
    stochastic gradient descent.
    Future TODO: Selection of alpha may be better performed via
    GridSearchCV. To quote sklearns documentation: 'Finding a reasonable
    regularization term is best done using GridSearchCV, usually in the range
    10.0**-np.arange(1,7).'"""

    # set up the dataset: If I understand the sourcecode correctly, the
    # SGDclassifier wants to have unique labels in a sample attribute
    # called 'targets' and is quite stubborn with this name - I could not convince
    # it to look for targets somewhere else, so now I'm catering to his demands
    if bilateral:
        ds.sa['targets'] = ds.sa.bilat_ROIs
    else:
        ds.sa['targets'] = ds.sa.all_ROIs

    clf = mv.SKLLearnerAdapter(SGDClassifier(loss='hinge',
                                            penalty='l2',
                                            class_weight='balanced'))

    cv = mv.CrossValidation(clf, mv.NFoldPartitioner(attr='participant'),
                            errorfx=mv.mean_match_accuracy,
                            enable_ca=['stats'])

    results = cv(ds)

    # save classification results
    with open(results_dir + 'SGD_clf.txt', 'a') as f:
        f.write(cv.ca.stats.as_string(description=True))

    if bilateral:
        desired_order = ['brain', 'VIS', 'LOC', 'OFA', 'FFA', 'EBA', 'PPA']
    else:
        desired_order = ['brain', 'VIS', 'left LOC', 'right LOC',
                         'left OFA', 'right OFA', 'left FFA',
                         'right FFA', 'left EBA', 'right EBA',
                         'left PPA', 'right PPA']

    labels = get_known_labels(desired_order,
                              cv.ca.stats.labels)

    # print confusion matrix with pymvpas build in function
    cv.ca.stats.plot(labels=labels,
                     numbers=True,
                     cmap='gist_heat_r')
    plt.savefig(results_dir + 'confusion_matrix.png')

    # print confusion matrix with matplotlib
    if niceplot:
        ACC = cv.ca.stats.stats['mean(ACC)']
        plot_confusion(cv,
                       labels,
                       fn=results_dir + 'confusion_matrix_SGD.svg',
                       figsize=(9, 9),
                       vmax=100,
                       cmap='Blues',
                       ACC='%.2f' % ACC)

    mv.h5save(results_dir + 'SGD_cv_classification_results.hdf5', results)
    print('Saved the crossvalidation results.')

    return cv



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--inputfile', help="An hdf5 file of the avmovie"
                        "or localizer data with functional ROI information,"
                        "transposed",
                        required=True)
    parser.add_argument('-bi', '--bilateral', help="If false, computation will "
                        "be made on hemisphere-specific ROIs (i.e. left FFA, "
                        "right FFA", default=True)
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
    parser.add_argument('-n', '--niceplot', help="If true, the confusion matrix of the "
                        "classification will be plotted with Matplotlib instead of build "
                        "in functions of pymvpa.", default=False)

    args = parser.parse_args()

    # get the data
    ds_file = args.inputfile
    ds = mv.h5load(ds_file)

    results_dir = '/' + args.output + '/'
    # create the output dir if it doesn't exist
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    ds_type = args.dataset
    bilateral = args.bilateral
    niceplot = args.niceplot
    coords = args.coords

    # strip potential overlaps from the brain
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
    # or append coordinates and get rid of fmri data is specified
    elif coords == 'only-coordinates':
        ds = get_voxel_coords(ds,
                              append=False,
                              zscore=False)

    sensitivities, cv = dotheclassification(ds,
                                            bilateral=bilateral)
