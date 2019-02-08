# localizer

This Github repository contains the scripts - and in a later, final stage also results - for a novel approach at determining the specificity of functional ROIs. The method in questions works by regressing the sensitivities derived from a leave-one-out crossvalidation classification with sensitivity analysis of any two ROIs in question onto the available experimental design to obtain a functional description of it.

The directory sourcedata contains submodules with the various input files:
- studyforrest phase 2 avmovie data
- studyforrest movie annotation data
- studyforrest visualrois/objectcategory data
- custom additional automatic movie annotation
- templates and transformations for the studyforrest dataset

Each subjects subdirectory contains per session (localizer, movie) custom preprocessed files. The preprocessing of thefiels was done with a nipype workflow that can be found in the code directory as preprocess_locdata.py

The derivatives directory contains 
- groupdatasets/ (large groupdatasets containing the localizer or moviedata together with ROI information, produced with the scripts code/create_hdf5_avmovie.py and code/create_hdf5_localizer.py)
- stimuli/detected_faces_events.tsv (automatically extracted faces per frame for all runs)
- stimuli/researchcut/avmovie (the subsets of this large file corresponding to each movie run. This cutting was done with sourcedata/annotations/code/researchcut2segments.py)
- stimuli/researchcut/downsampled_event_files (The former event files as proper event files (with code/faces2events.py), and downsampled to seconds (with code/downsample_face_events.py)
- results: Results of various flavours of the analysis on either the location data (performed with code/localizer_cv_glm_analysis.py) or the moviedata (performed with code/avmovie_cv_glm_analysis.py).
- groupspace_analysis: The same analysis as in results, but performed in
  groupspace. This is the relevant directory to look for results. 

The classification and glm analysis scripts contain commandline-specifyable options to compute partial results (classification, glm, plotting) on the full or only-ROI dataset, containing either no coordinates, additional coordinate information, or only coordinate information.
The create_hdf5 scripts contain commandline-specifyable options to zscore normally, not at all, or - in the case of the localizer data - on parameters derived from rest periods with no stimulation.

NOTE: all of this here is work in progress.
