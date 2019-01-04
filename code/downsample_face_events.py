#!/usr/bin/env python

""" takes the segmented event files and downsamples them. 
Specify path to the segmented event files and path where to store outputs. """

import pandas as pd
from glob import glob
import os.path

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help="specify directory where event files are located", required=True)
    parser.add_argument('-o', '--output', help="specify directory where finished event files should go", required=True)
    args = parser.parse_args()
# read the datafiles from specificied directory

files = sorted(glob(args.input + '/*'))
assert len(files) == 8

# downsample the files
for file in files:
    data = pd.read_csv(file, sep='\t')
    # round to full seconds
    data['onset'] = data.onset.apply(round, ndigits=0)
    duration = data.groupby('onset').sum().duration
    amplitude = data.groupby('onset').max()['condition']
#    condition =  ['face' if i > 0 else 'no_face' for i in amplitude]
    condition =  ['no_face' if i == 0 else 'face' if i <= 3 else 'many_faces' for i in amplitude]
    onset = [float(i) for i in range(0, len(duration))]
    df = pd.DataFrame({'onset': onset, 'duration': duration, 'condition': condition})
    cols = ['onset', 'duration', 'condition']
    df = df[cols]
  ##  data['bool_faces'] = [1 if i > 0 else 0 for i in data['condition']]
  ##  data['onset'] = [float(i) for i in range(0, len(data))]
  ##  data.rename(columns={'condition':'total_faces'}, inplace=True)
    # save result
    if not os.path.isdir(args.output):
        os.makedirs(args.output)
    df.to_csv(args.output + '/' + os.path.split(file)[1], sep = '\t', header=True, index=False)

