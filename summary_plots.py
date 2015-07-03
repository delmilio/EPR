import os, pylab
import numpy as np
import matplotlib.pyplot as plt


# ==============================
#
# USAGE INSTRUCTIONS
#
# Execute this script via the command line:
# python summary_plots.py
# It will automatically go through ALL of the folders in the current working
# directory, pick out the statistics_analysis.txt, and create the plots.
# The x-axis is labeled with the name of the folder.
# Folder names must be in the form "sample 0", "sample 1", etc.
#
# ==============================


def get_sample_names():
    """
    Gets the folder names.
    """
    return [name for name in os.listdir(".") if os.path.isdir(name)]


def get_sample_data():
    """
    Return format: {"sample 0": [amplitude, amplitude st dev, peak distance, peak distance st dev, peak separation, peak separation st dev]}
    """
    return [extract_statistics_data(open("sample " + str(sample) + "/statistics_analysis.txt").readlines()) for sample in sorted(map(lambda sample: int(sample.split()[-1]), get_sample_names()))]


def extract_statistics_data(sample_data):
    """
    Read the statistics data
    """
    data = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for line in sample_data:
        if "Mean amplitude (V)" in line:
            data[0] = float(line.split()[-1])
        if "Amplitude standard deviation (V)" in line:
            data[1] = float(line.split()[-1])
        if "Mean max amplitude current (A)" in line:
            data[2] = float(line.split()[-1])
        if "Max amplitude current standard deviation (A)" in line:
            data[3] = float(line.split()[-1])
        if "Mean min amplitude current (A)" in line:
            data[4] = float(line.split()[-1])
        if "Min amplitude current standard deviation (A)" in line:
            data[5] = float(line.split()[-1])
        if "Mean peak distance (V)" in line:
            data[6] = float(line.split()[-1])
        if "Peak distance standard deviation (V)" in line:
            data[7] = float(line.split()[-1])
        if "Mean peak separation (A)" in line:
            data[8] = float(line.split()[-1])
        if "Peak separation standard deviation (A)" in line:
            data[9] = float(line.split()[-1])
    return data


def error_plot(measurement):
    """
    Plot of amplitude versus sample number, showing an error of one standard
    deviation.

    measurement can be:
    0: amplitude
    2: max amplitude current
    4: min amplitude current
    6: peak distance
    8: peak separation
    """
    fig = pylab.figure()
    measurement_title = {0: "Amplitude (V)", 2: "Max amplitude current (A)", 4: "Min amplitude current (A)", 6: "Peak distance (V)", 8: "Peak separation (A)"}
    print "Processing " + measurement_title[measurement]
    values = [data[measurement] for data in get_sample_data()]
    errors = [data[measurement + 1] for data in get_sample_data()]
    plt.errorbar(range(len(values)), values, yerr=errors)
    for (label, x, y) in zip(get_sample_names(), range(len(values)), values):
        plt.annotate(label, xy=(x, y), xytext=(5, 5), textcoords="offset points")
    pylab.title(measurement_title[measurement])
    pylab.xlabel("Sample")
    pylab.ylabel("Value and error (one standard deviation)")
    pylab.xlim([-1, len(values)])
    pylab.savefig(measurement_title[measurement])
    # plt.show()


for i in [0, 2, 4, 6, 8]:
    error_plot(i)
