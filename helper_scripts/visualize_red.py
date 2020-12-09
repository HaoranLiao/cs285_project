'''
This script provides learning curves visualization for the RED experiments
'''

import sys
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import re
import pandas as pd

visualization_items = ["Best so far", "entloss", "entropy", "EpisodesSoFar", "EpLenMean", "EpMeanReturn(100 ep)", "EpRewMean", "EpThisIter", "EpTrueRewMean", "ev_tdlam_before", "meankl", "optimgain", "surrgain"]

def get_values(log_file):
    return_dict = {item: [] for item in visualization_items}
    timesteps = []
    with open(log_file) as f:
        for line in f:
            for item in visualization_items:
                if item in line:
                    num = float(line.split("|")[-2])
                    return_dict[item].append(num)
            if "TimestepsSoFar" in line:
                num = int(line.split("|")[-2])
                timesteps.append(num)
    df = pd.DataFrame.from_dict(return_dict)
    df.index = timesteps
    return df

def plot(dataframe, plotting_categories):
    plt.figure()
    for cat in plotting_categories:
        plt.plot(dataframe.index, dataframe[cat], label=cat)
    plt.legend()
    plt.xlabel("Timesteps")
    plt.ylabel("Value")
    plt.show()

df = get_values(sys.argv[1])
print("Index   |  Column name")
print("0       |  All")
for idx, col_name in enumerate(visualization_items):
    print("%-8d|  %s" % (idx + 1, col_name))
while True:
    inp = input("What to visualize?")
    if inp == "0":
        plotting_categories = visualization_items
    else:
        plotting_categories = [visualization_items[int(i) - 1] for i in inp.split(" ")]
    plot(df, plotting_categories)
