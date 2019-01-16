#!/usr/bin/env python

import mvpa2.suite as mv
import numpy as np
import pandas as pd



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
            assert 'brain' in ds.sa.all_ROIs
            print('excluded the rest of the brain from the dataset.')
        if 'overlap' in np.unique(ds.sa.all_ROIs):
            ds = ds[(ds.sa.all_ROIs != 'overlap'), :]
            assert 'overlap' in ds.sa.all_ROIs
    if order == 'sparse':
        print("attempting to exclude any overlaps from the dataset.")
        if 'overlap' in np.unique(ds.sa.all_ROIs):
            ds = ds[(ds.sa.all_ROIs != 'overlap'), :]
            assert 'overlap' in ds.sa.all_ROIs
            print('excluded overlap from the dataset.')
    return ds


def bilateralize(ds):
    """combine lateralized ROIs in a dataset."""
    ds_ROIs = ds.copy('deep')
    ds_ROIs.sa['bilat_ROIs'] = [label.split(' ')[-1] for label in ds_ROIs.sa.all_ROIs]
    mv.h5save(results_dir + 'ds_ROIs.hdf5', ds_ROIs)
    print('Combined lateralized ROIs for the provided dataset and saved the dataset.')
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
        classifications = []
        if bilateral:
            def store_class(data, node, result):
                class_ds = mv.Dataset(samples=data.sa.voxel_indices)
                class_ds.sa['targets']=data.sa.bilat_ROIs
                class_ds.sa['predictions']=clf.predict(data)
                class_ds.sa['LOsubj']=data.sa.participant
                classifications.append(class_ds)
            # TODO: think of a way to add information about the partition into the dataset.
        else:
            def store_class(data, node, result):
                class_ds = mv.Dataset(samples=data.sa.voxel_indices)
                class_ds.sa['targets']=data.sa.all_ROIs
                class_ds.sa['predictions'] = clf.predict(data)
                class_ds.sa['LOsubj'] = data.sa.participant
                classifications.append(class_ds)

        # do a crossvalidation classification and store the classification results
        cv = mv.CrossValidation(clf, mv.NFoldPartitioner(attr='participant'),
                                errorfx=mv.mean_match_accuracy,
                                enable_ca=['stats'],
                                callback=store_class)

        results = cv(ds)
        # save classification results as a Dataset
        ds_type = ['movie', 'loc']
        mv.h5save(results_dir + 'cv_classification_results_{}.hdf5'.format(ds_type[idx]), classifications)
        print('Saved the classification results obtained during crossvalidation.')

        # get the classification list into a pandas dataframe

        for i, classification in enumerate(classifications):
            df = pd.DataFrame(data={'voxel_indices': list(classification.samples),
                                    'targets': list(classification.sa.targets),
                                    'predictions': list(classification.sa.predictions),
                                    'fold': [i] * len(classification.sa.predictions),
                                    'ds_type': [ds_type[idx]] * len(classification.sa.predictions)
                                    }
                              )
            dfs.append(df)
    all_classifications = pd.concat(dfs)
    return all_classifications



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
    ds_type = args.dataset                      # stripped --> no brain, no overlap,

    if args.bilateral == 'True' or args.bilateral == True:
        bilateral = True
    elif args.bilateral == 'False' or args.bilateral == False:
        bilateral = False  # True or False

    coords = args.coords                        # no-coords --> leave ds as is,
                                                # with-coords --> incl. coords,
                                                # only-coords --> only coords
    classifier = args.classifier                # gnb, sgd, l-sgd --> multiclassclassifier

    # fail early, if classifier is not appropriately specified.
    allowed_clfs = ['l-sgd', 'gnb']
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
        ds_loc =bilateralize(ds_loc)

    # append coordinates
    ds_movie = get_voxel_coords(ds_movie,
                                append=True,
                                zscore=True)
    ds_localizer = get_voxel_coords(ds_localizer,
                                    append=True,
                                    zscore=True)


