#!/usr/bin/python

import os
from glob import glob
from subprocess import call

"""
This script will transform the BOLD data from the movie session
into group space. It was executed with datalad run in commit 
ae485fa14d02b9afcc66b410d4c17581f2d5bf52

Warning to myself: In an attempt to restructure and clean up everything
in a more BIDS-like fashing, I'm adding the ses-movie directory I previously
forgot to create. Once done, I need to make sure everything is still running
neatly.
"""


base_dir='/data/movieloc/backup_store/saccs/'
template_dir='/ses-movie/xfm/'
data_dir='/ses-movie/func/'
roi_dir='/ses-movie/anat/'

interpolation = 'nn'

participants = sorted([path.split('/')[-1] for path in glob(base_dir + 'sub-*')])

for participant in participants:
	input_fns = glob(base_dir + participant + data_dir + 'sub-*_task-avmovie_run-*_bold.nii.gz')
	input = [fn for fn in (glob(base_dir + participant + roi_dir + '*_mask.nii.gz')) if not os.path.basename(fn).startswith('brain')]
	input_fns.extend(input)
	input_fns.extend(glob(base_dir + participant + data_dir + 'sub-*highpass*'))
	input_fns.append(base_dir + participant + roi_dir + 'brain_mask.nii.gz')

	reference_fn = base_dir + participant + template_dir + 'NonstandardReference_space-group.nii.gz'
	warp_fn = base_dir + participant + template_dir + participant + '_from-BOLD_to-group_mode-image.nii.gz'

	output_dir = base_dir + participant 
	

	for input_fn in input_fns:
		#save tmpl.mask files in roi_dir
		if input_fn.split('/')[-1].split('.')[0].endswith('mask'):
			output_fn = output_dir + roi_dir + input_fn.split('/')[-1].split('.')[0] + '_tmpl.nii.gz' 
		#save tmpl.bold files in data_dir
		else:
			output_fn = output_dir + data_dir + input_fn.split('/')[-1].split('.')[0] + '_tmpl.nii.gz'
	
		fsl_cmd = ("fsl5.0-applywarp -i {0} -o {1} -r {2} -w {3} --interp={4}".format(
                input_fn, output_fn, reference_fn, warp_fn, interpolation))

		call(fsl_cmd, shell=True)
		print("Warped input file {0}; output file is {1}".format(
                    input_fn, output_fn))

