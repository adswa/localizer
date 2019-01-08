#!/usr/bin/env python

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle



"""
A script to compute a plot of variable co-occurrence of events in the avmovie
dataset. Specify the full event file and state up to 9 variables which cooccurance 
you would want to check.

"""

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-v", "--variables", nargs='+', help="Which variables to build contingency tables from," \
                        "supply as consecutive strings in command line as 'face' 'many_faces' 'scene-change'")
    parser.add_argument("-e", "--eventfile", help="Path to the full event file ('full_event_file.tsv') that "
                                                  "is saved during glm computation")
    parser.add_argument("-s", "--save", help="Save the figure (True/False), default: False", default=False)

    args = parser.parse_args()

    variables = [i for i in args.variables]
    if len(variables)>9:
        print("I can only print 9 variables at once. Anything past variable 9 will not be shown")
    event_file = args.eventfile
    save = args.save

def plotvarasrectangle(x, y, color):
    """plots a variable rectangle of 6 seconds, given offset variable as x and a value for y"""
    ax.add_patch(Rectangle((x - 3, y), 6, 0.1, facecolor=color, alpha=0.3))

#read in data
data = pd.read_csv(event_file, sep='\t')
movie_len = max(data.onset)
# we should round, else faces onsets (up to like.. 10 decimals precise) does not
# match other onsets (up to 2 decimals precise)
data = data.round(0)

# dummy vector of length movie time
movie_time = np.arange(0, movie_len)

# start a figure
fig, ax = plt.subplots(figsize=(100, 2))
plt.xlim(0, movie_len)

colors = ['red', 'blue', 'green', 'brown', 'purple', 'orange', 'black', 'pink', 'darkblue']
# initialize height for variable rectangle
y = 0

# this makes a plot
for variable in variables:
    color = colors[0]
    onset_time = data[data.condition==variable]['onset'].values
    offset_time = data[data.condition==variable]['onset'].values + \
                  data[data.condition==variable]['duration'].values
    assert len(onset_time) == len(offset_time)
    # get a '1' for every movietime point containing the variable offset
    var = [1 if i in offset_time else 0 for i in movie_time]
    # for every new variable, increase the height of plotted rectangle
    y += 0.1
    for idx, i in enumerate(var):
        if i == 1:
            plotvarasrectangle(movie_time[idx], y, color)
    del colors[0]
plt.title("in ascending order:" + str([str(i) for i in variables]))
if save == True:
    plt.savefig('cooccurrence_plot.png')
else:
    plt.show()
