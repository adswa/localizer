#!/home/adina/env/wtf3/bin/python
"""
This script will get an rudimentary event file from face detection json files.
"""
import json
import gzip

base_dir = '/data/movieloc/backup_store/saccs/'
source_dir = '/sourcedata/'


# load in the data from json
facefile = base_dir + source_dir + 'comp_stim_features/data/faces/detected_faces.json*'
with gzip.open(facefile, 'rt') as f:
    json_data = json.load(f)

# put the data into a dataframe, corresponding to the layout of an event file
df = pd.DataFrame([{k: j[k] for k in ('onset', '#faces')} for i, j in enumerate(json_data)])
df.columns = ['condition', 'onset']
# get durations of 40ms in there
df['duration'] = 0.04

# save the data file

df.to_csv(base_dir + source_dir + 'comp_stim_features/data/faces/events.tsv',
          sep = '\t',
          header = True,
          index = False)

