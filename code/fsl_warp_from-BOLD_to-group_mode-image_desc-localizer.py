#!/usr/bin/python

from glob import glob
from subprocess import call

#Transform localizer data from subject to group space.
#We could get each subjects masks also into the ses-localizer
#subdirectory, but its essentially the same as in movie...
#

base_dir='/data/movieloc/backup_store/saccs/'
template_dir='/ses-movie/xfm/'
data_dir='/ses-localizer/func/'
roi_dir='/ses-movie/anat/'

interpolation = 'nn'

participants = sorted([path.split('/')[-1] for path in glob(base_dir + 'sub-*')])
for participant in participants:
    input_fns = glob(base_dir + participant + data_dir +
                     'sub-*_task-objectcategories_run-*_space-custom-subject_desc-highpass_bold.nii.gz')
    reference_fn = base_dir + participant + template_dir + 'NonstandardReference_space-group.nii.gz'
    warp_fn = base_dir + participant + template_dir + participant + '_from-BOLD_to-group_mode-image.nii.gz'

    output_dir = base_dir + participant

    for input_fn in input_fns:
        # save tmpl.bold files in data_dir
 ##       output_fn = output_dir + data_dir + input_fn.split('.')[0] + '_tmpl.nii.gz'
        output_fn = output_dir + data_dir + input_fn.split('/')[-1].split('.')[0] + '_tmpl.nii.gz'

        fsl_cmd = ("fsl5.0-applywarp -i {0} -o {1} -r {2} -w {3} --interp={4}".format(
            input_fn, output_fn, reference_fn, warp_fn, interpolation))
        call(fsl_cmd, shell=True)
        print("Warped input file {0}; output file is {1}".format(input_fn, output_fn))
