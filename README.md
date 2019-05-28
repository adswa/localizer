# Turn it sideways

This Github repository contains the scripts and preliminary results for a
classification-based approach at determining the specificity of functional ROIs.

## Method summary

The method in questions works by 1) deriving sensitivities during classification analysis in
a leave-one-participant-out crossvalidation of any two ROIs, and 2)
modelling this time course of sensitivities with the available experimental design to
provide a functional description of it.
Comparing this approach with the typical "standard" univariate GLM with canonical
contrasts on the fMRI data (e.g. faces versus houses) shows that
1) such canonical contrasts may not work well for complex stimulation
2) the functional specificity of ROIs derived from simplistic designs is diverse during complex stimulation

Check out the poster in ``derivatives/poster`` for a graphical description of analyses and results.

## Repo overview

The repository is a [datalad dataset](datalad.org). In theory, the commit history
should contain all relevant information about the origin of source data, its preprocessing,
the performed analyses and the corresponding results. Nevertheless, the history
is long, and scripts underwent constant refactoring. Therefore, this README tries
to help by giving an overview of the contents of this repository.

### Sourcedata

The directory ``sourcedata/`` contains submodules with the various input files (all linked as [datalad](datalad.org)
subdatasets):

- [studyforrest phase 2 movie + block design data](https://github.com/psychoinformatics-de/studyforrest-data-phase2)
- [studyforrest movie annotation data](https://github.com/psychoinformatics-de/studyforrest-data-annotations)
- [studyforrest visualrois data](https://github.com/psychoinformatics-de/studyforrest-data-visualrois)
- custom additional automatic movie annotation with pliers (McNamara et al., 2017), (data yet to be published)
- [templates and transformations for the studyforrest dataset](https://github.com/psychoinformatics-de/studyforrest-data-templatetransforms)

### Code

The directory ``code/`` contains all scripts used for the analyses. The important ones are:
- ``fsl_warp*``: warps images from subject- to groupspace
- ``preproc*``: preprocessing workflow
- ``create_hdf5_ds``: creates groupdatasets from all subjects
- ``cv_clf_glm_main.py``: main analysis script for all analyses
- ``utils.py``: Helper function for the main analyses
- ``*face*``: Scripts to preprocesses face annotations

The main analysis script contains commandline-specifyable
options to compute partial results (classification, glm, plotting) on the
full or only-ROI dataset, containing either no coordinates, additional
coordinate information, or only coordinate information, with different
classifiers (GLM, SGD, l-SGD), and to reverse the analysis.
The ``create_hdf5_ds.py`` script contain commandline-specifyable options to zscore
normally, not at all, or - in the case of the localizer data - on parameters
derived from rest periods with no stimulation.


### sub-* directories

Each subjects subdirectory contains -  per session (block-design, movie) -
custom preprocessed files in subject and group space.
Preprocessing was done with a nipype workflow
that can be found in the code directory as ``code/preprocess_locdata.py``.
The script was executed with ``datalad-containers run`` in a singularity
image (https://www.singularity-hub.org/collections/1877)
Subject space files were warped into a study-specific group space based on templates
in the studyforrest dataset, using FSL and the script
``code/fsl_warp_from-BOLD_to-group_mode-image.py``.

### Derivatives

The ``derivatives/`` directory contains
- ``groupdatasets/``: large groupdatasets in subject space containing the block design or moviedata
  together with ROI information, produced with the script ``code/create_hdf5_ds.py``
- ``ds_groupspace/``: Similar groupdatasets, but in group space. **These files are the analysis base.**
- ``stimuli/detected_faces_events.tsv``: automatically extracted faces per
  frame for all runs, based on
- ``stimuli/researchcut/avmovie``: the subsets of this large file
  corresponding to each movie run. This cutting was done with
  ``sourcedata/annotations/code/researchcut2segments.py``)
- ``stimuli/researchcut/downsampled_event_files``: The former event files as
  proper event files (with ``code/faces2events.py``), and downsampled to
  seconds (with ``code/downsample_face_events.py``)
- ``results/``: [OUTDATED -- IGNORE] Results of various flavours of the analysis on either the location data (performed with code/localizer_cv_glm_analysis.py) or the moviedata (performed with code/avmovie_cv_glm_analysis.py).
- ``groupspace_analysis/``: Analyses results performed in groupspace for various
flavours (classification --> GLM/GLM --> classification; GNB/SGD/l-SGD; FFA-vs-PPA/FFA-vs-brain;
normalized results/unnormalized results; movie/block-design data; Full dataset/stripped
dataset). This is the relevant directory to look for results.
- ``poster/``: contains the corresponding poster presented at the OHBM in Rome


NOTE: all of this here is work in progress. Refactoring happens quite often.


## Acknowledgements
The analyses rely on open source software (nipype, numpy, pandas, scipy, sklearn, seaborn,
matplotlib, pylab, datalad, git, git-annex, FSL, Neurodebian, PyMVPA, pliers, and likely many more).
We want to thank the authors, contributers and maintainers of these awesome software
projects for making it possible. The analyses further rely on shared, open data. We want to
thank the authors and anonymous subjects of these datasets.