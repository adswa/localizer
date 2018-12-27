import pandas as pd
import numpy as np
import seaborn as sns

"""
tiny script to explore role of temporal progression and occurance of faces in the
avmovie glm data, bc time+ and time- have reliably strong coefficients in results.
best to be run in an interactive environment. Takes the full_event_file of the
avmovie glm analysis.
"""

#read in data
data = pd.read_csv('full_event_file.tsv', sep='\t')

# we should round, else faces onsets (up to like.. 10 decimals precise) does not
# match other onsets (up to 2 decimals precise)
data = data.round(0)

# its a bit tricky (or I'm dumb, also likely) to get to the contigency table
# based on the event file. I'm going to do a detour...

# get movielength in seconds.
movie_len = max(data.onset)
# for each condition, constructs arrays with a 1 whenever the condition is
# fullfilled at a given timepoint, else leave it zero. For this, extract onsets
# of events
onsets_time_plus = data[data.condition=='time+']['onset'].round(0).values
onsets_time_minus = data[data.condition=='time-']['onset'].round(0).values
onsets_time_manyfaces = data[data.condition=='many_faces']['onset'].round(0).values
onsets_time_face = data[data.condition=='face']['onset'].round(0).values

movie_time = np.arange(0, movie_len)
faces = [1 if i in onsets_time_face else 0 for i in movie_time]
manyfaces = [1 if i in onsets_time_manyfaces else 0 for i in movie_time]
time_plus = [1 if i in onsets_time_plus else 0 for i in movie_time]
time_minus = [1 if i in onsets_time_minus else 0 for i in movie_time]
#categories face and manyface are mutually exclusive:
facefull = np.asarray['face' if faces[i]==1 else 'manyface' if manyfaces[i]==1 else
'no_face' for i in range(len(movie_time))]
times = np.asarray['past' if time_minus[i]==1 else 'future' if time_plus[i]==1 else
'constant' for i in range(len(movie_time))]

#contigency table:
c = pd.crosstab(times, facefull, normalize='index')

#to plot:
s = c.stack().reset_index().rename(columns={0: 'value'})
sns.barplot(x=s.row_0, y=s.value, hue=s.col_0)

