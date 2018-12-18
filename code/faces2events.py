#!/home/adina/env/wtf3/bin/python
"""
This script will get an rudimentary event file from face detection json files.
Supply json file with --infile flag and output with --outfile path
"""
import json
import gzip
import pandas as pd

if __name__ == '__main__':
    import argparse

    parser =  argparse.ArgumentParser()
    parser.add_argument('-i', '--infile', help = 'json.gz file with detected faces', required = True)
    parser.add_argument('-o', '--outfile', help = 'path and filename where output should go to', required = True)

    args = parser.parse_args()

    facefile = args.infile
    outfile = args.outfile

    print(facefile, outfile)
# load in the data from json
with gzip.open(facefile, 'rt') as f:
    json_data = json.load(f)

# put the data into a dataframe, corresponding to the layout of an event file
df = pd.DataFrame([{k: j[k] for k in ('onset', '#faces')} for i, j in enumerate(json_data)])
df.columns = ['condition', 'onset']
# get durations of 40ms in there
df['duration'] = 0.04

# save the data file

df.to_csv(outfile,
          sep = '\t',
          header = True,
          index = False)


