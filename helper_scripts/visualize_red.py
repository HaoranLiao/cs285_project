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

def plot_multiple_run(dataframes, plotting_category):
    plt.figure()
    for df_name in dataframes:
        plt.plot(dataframes[df_name].index, dataframes[df_name][plotting_category], label=df_name)
    plt.legend()
    plt.xlabel("Timesteps")
    plt.ylabel("Value")
    plt.title(plotting_category)
    plt.show()


dfs = {}
for item in sys.argv[1:]:
    dfs[os.path.basename(item)] = get_values(item)

print("Index   |  Column name")
print("0       |  All")
for idx, col_name in enumerate(visualization_items):
    print("%-8d|  %s" % (idx + 1, col_name))
while True:
    inp = input("What to visualize?")
    if inp == "0":
        assert len(dfs) == 1
        plotting_categories = visualization_items
        plot(dfs.values()[0], plotting_categories)
    elif inp == "666":
        for df_name in dfs:
            dfs[df_name]["EpMeanReturn(100 ep)"].to_csv(path_or_buf='out.csv')
    else:
        plotting_category = visualization_items[int(inp) - 1] 
        plot_multiple_run(dfs, plotting_category)
    
