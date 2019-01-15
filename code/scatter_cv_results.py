#!/usr/bin/env python

import mvpa2.suite as mv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

"""
This script will make a scatterplot from the confusion 
matrices of two classification analysis on the same targets.
"""

# we need to get to the classification
def dotheclassification(ds,
                        classifier,
                        bilateral
                        ):
    """ Dotheclassification does the classification.
    Input: the dataset on which to perform a leave-one-out crossvalidation with a classifier
    of choice.
    Specify: the classifier to be used (gnb (linear gnb), l-sgd (linear sgd), sgd)
             whether the sensitivities should be computed and stored for later use
             whether the dataset has ROIs combined across hemisphere (bilateral)
    """
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


    cv = mv.CrossValidation(clf, mv.NFoldPartitioner(attr='participant'),
                            errorfx=mv.mean_match_accuracy,
                            enable_ca=['stats'])
    results = cv(ds)
    return cv


def main(ds_1,
         ds_2,
         bilateral,
         clf,
         output=None):
    # lets get the trained classifier
    cv_1 = dotheclassification(ds_1,
                               bilateral=bilateral,
                               classifier=clf,
                               )
    cv_2 = dotheclassification(ds_2,
                               bilateral=bilateral,
                               classifier=clf,
                               )

    # make sure the datasets were in the same space, i.e. the total number
    # of voxel per ROI is the same
    import numpy.testing as nt
    nt.assert_array_equal(np.sum(cv_1.ca.stats.matrix, axis=0), np.sum(cv_2.ca.stats.matrix, axis=0))
    cv_1_matrix_string = cv_1.ca.stats.matrix.flatten()
    cv_2_matrix_string = cv_2.ca.stats.matrix.flatten()

    # lets define colors based on the size of the matrix, as the classifier:
    labels = cv_1.ca.stats.labels
    cycol = ['orangered', 'chartreuse', 'b', 'gold', 'deepskyblue', 'black', 'silver']

    custom_legend = []
    for i in range(len(labels)):
        custom_legend.append(Line2D([0], [0], marker='o', label=labels[i], color='w',
                                    markerfacecolor=cycol[i], markersize=15))

    fig, ax = plt.subplots()
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: "%.0f" % (np.exp(y))))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: "%.0f" % (np.exp(x))))
    axis_text = "voxel count in confusion matrix"
    ax.set_xlabel(axis_text + ' 1')
    ax.set_ylabel(axis_text + ' 2')

    for idx_row, lab_row in enumerate(labels):
        for idx_col, lab_col in enumerate(labels):
            # prediction label defines outer color
            edgecolor = cycol[idx_row]
            # target label defines fill
            facecolor = cycol[idx_col]
            plt.scatter(np.log(cv_1.ca.stats.matrix[idx_row,idx_col]),
                        np.log(cv_2.ca.stats.matrix[idx_row, idx_col]),
                        s = 80,
                        facecolor=facecolor,
                        edgecolor=edgecolor)
    ax.legend(handles=custom_legend)
    ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
    plt.title('Scatterplot of the cells of two confusion matrices')
    if output:
        plt.savefig(output + 'Scatterplot_confusionmatrices.png')
    else:
        plt.savefig('scatterplot_confusionmatrices.png')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input1', help="Give me a path to "
                                               "the first dataset "
                                               "(movie or localizer)")
    parser.add_argument('-j', '--input2', help="Give me a path to "
                                               "the second dataset "
                                               "(movie or localizer)")
    parser.add_argument('-c', '--classifier', help="Which classifier"
                                                   "to use: gnb or"
                                                   "l-sgd")
    parser.add_argument('-b', '--bilateral', help="Is the dataset"
                                                  "lateralized or do we"
                                                  "have combined ROIs?",
                        default=True)
    parser.add_argument('-o', '--output', help="Where to save fig?")
    args = parser.parse_args()

    # load the data
    ds_1_path = args.input1
    ds_2_path = args.input2
    ds_1 = mv.h5load(ds_1_path)
    ds_2 = mv.h5load(ds_2_path)

    # check ds status
    if args.bilateral:
        bilateral = True
    else:
        bilateral = False

    if args.output:
        output = args.output
    else:
        output = None

    # get the classifier information
    valid_clfs = ['gnb', 'l-sgd']
    clf = args.classifier
    assert clf in valid_clfs

    # check what dataset size we had been given:
    if bilateral:
        if 'brain' in ds_1.sa.bilat_ROIs:
            ds_type = 'full'
        else:
            ds_type = 'stripped'
    else:
        if 'brain' in ds_1.sa.all_ROIs:
            ds_type = 'full'
        else:
            ds_type = 'stripped'

    main(ds_1,
         ds_2,
         bilateral=bilateral,
         clf=clf,
         output=output)


