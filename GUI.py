from Tkinter import *
from collections import defaultdict
import os, datetime, time, numpy, shutil, tkFileDialog

import matplotlib
matplotlib.use('TkAgg')

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt


class TrialSummary():
    """
    Similar to the Trial class. However, is only to be used if statistics_analysis.txt already exists for this trial.
    The purpose of this class is to provide an interface for loading information about the trial without the overhead
    needed to compute it again.
    """
    def __init__(self, directory):
        # Find Data Directory
        self.data_directory = directory

        """
        For parsing comment.txt and statistics_analysis.txt it is assumed that the data follows the same format that
        this program outputs statistics_analysis.txt and comment.txt
        """

        # Find Comment File and Extract Information
        self.comment_file = open(os.path.join(self.data_directory, 'comment.txt')).read().split('\n')
        self.name = self.comment_file[0][8:]
        self.min_current = float(self.comment_file[1][13:-2])
        self.max_current = float(self.comment_file[2][13:-2])
        self.sensitivity = float(self.comment_file[5][13:-3])

        # Find Statistics File and Extract Information
        self.stats_file = open(os.path.join(self.data_directory, 'statistics_analysis.txt')).read().split('\n')
        summary_start_loc = self.stats_file.index('RESULTS SUMMARY') + 1
        self.stats_file = self.stats_file[summary_start_loc:]

        self.stats_list = []
        for stat in self.stats_file:
            if stat != '':
                i = stat.index(':') + 1
                self.stats_list.append(float(stat[i:]))

        # Load Average Response Data
        average_response = open(os.path.join(self.data_directory, 'average_signal_data.txt')).read().split('\n')
        average_response.pop()  # Remove last empty element

        self.average_current_data = []
        self.average_voltage_data = []
        for point in average_response:
            data = point.split('\t')
            self.average_current_data.append(float(data[1]))
            self.average_voltage_data.append(float(data[2]))

    def get_average_response(self):
        """
        returns the average response normalized to 1mv sensitivity
        """
        return self.average_current_data, [v*self.sensitivity for v in self.average_voltage_data]

    def get_stats_list(self):
        """
        return stats list normalized to 1mv sensitivity
        """
        normalized_stats = self.stats_list
        for i in [0, 1, 6, 7]:
            normalized_stats[i] *= self.sensitivity
        return normalized_stats

    def save_data(self, new_dir):
        with open(os.path.join(new_dir, 'statistics_analysis.txt'), 'a') as text_file:
            text_file.write('RESULTS SUMMARY' + ' -- Sample ' + self.name + '\n')
            text_file.write('Sensitivity: ' + str(self.sensitivity) + '\n')

            text_file.write('Mean amplitude (V): ' + str(self.stats_list[0]) + '\n')
            text_file.write('Amplitude standard deviation (V): ' + str(self.stats_list[1]) + '\n\n')

            text_file.write('Mean max amplitude current (A): ' + str(self.stats_list[2]) + '\n')
            text_file.write('Max amplitude current standard deviation (A): ' + str(self.stats_list[3]) + '\n\n')

            text_file.write('Mean min amplitude current (A): ' + str(self.stats_list[4]) + '\n')
            text_file.write('Min amplitude current standard deviation (A): ' + str(self.stats_list[5]) + '\n\n')

            text_file.write('Mean peak distance (V): ' + str(self.stats_list[6]) + '\n')
            text_file.write('Peak distance standard deviation (V): ' + str(self.stats_list[7]) + '\n\n')

            text_file.write('Mean peak separation (A): ' + str(self.stats_list[8]) + '\n')
            text_file.write('Peak separation standard deviation (A): ' + str(self.stats_list[9]) + '\n\n')

            text_file.write('Mean peak separation (number of samples): ' + str(self.stats_list[10]) + '\n')
            text_file.write('Peak separation standard deviation (number of samples): ' + str(self.stats_list[11]) + '\n\n\n')
        return


class Trial():

    def __init__(self, directory):

        # Find Data Directory
        self.data_directory = directory

        # Find Comment File and Extract Information
        self.comment_file = open(os.path.join(self.data_directory, 'comment.txt')).read().split('\n')
        self.name = self.comment_file[0][8:]
        self.min_current = float(self.comment_file[1][13:-2])
        self.max_current = float(self.comment_file[2][13:-2])
        self.sensitivity = float(self.comment_file[5][13:-3])

        # Find the Data Files
        data_files = [data_file for data_file in os.listdir(self.data_directory) if data_file.find('data_') >= 0 and data_file.find('.lvm') >= 0]
        self.data = []
        for data in data_files:
            self.data.append(Data(self, os.path.join(self.data_directory, data)))

        # Define vars for selecting current sample
        self.current_sample_index = 0

    def __str__(self):
        return self.name

    def current_sample(self):
        """
        Return the current sample's Data object
        """
        return self.data[self.current_sample_index]

    def select_next_sample(self):
        if self.current_sample_index + 1 < len(self.data):
            self.current_sample_index += 1
        else:
            self.current_sample_index = 0
        self.plot_data()

    def select_previous_sample(self):
        if self.current_sample_index - 1 < 0:
            self.current_sample_index = len(self.data) - 1
        else:
            self.current_sample_index -= 1
        self.plot_data()

    def linear_reg_model(self, n):
        m = (self.max_current - self.min_current) / 3.132982
        b = self.min_current + 0.211337*(self.max_current - self.min_current)/3.132982
        return n*m + b

    def plot_data(self):
        # Clear the plot
        plt.figure(1)
        plt.clf()

        # Plot all signals without bound lines in subplot 1
        plt.subplot(211)
        plt.grid(True)
        plt.xlim([self.min_current, self.max_current])
        for line in self.data:
            line.graph_trial_data()

        # Plot selected signal with bound lines in subplot 2
        plt.subplot(212)
        plt.grid(True)
        plt.xlim([self.min_current, self.max_current])
        self.current_sample().graph_trial_data(True)

        # Show the plot
        plt.figure(1).canvas.draw()
        return

    def save_data(self):
        # Create new directory and variables that point to it
        timestamp = str(datetime.datetime.fromtimestamp(time.time()).strftime('%m-%d-%Y %I-%M-%S %p'))
        new_dir = os.path.join(self.data_directory, timestamp)
        os.makedirs(new_dir)

        # Make stats file and copy over data files
        stats_file = os.path.join(new_dir, 'statistics_analysis.txt')
        data_stats = []
        for data in self.data:
            shutil.copyfile(data.path, os.path.join(new_dir, 'data_'+data.sample_num+'.lvm'))
            data_stats.append(data.statistics_data(stats_file))
        data_stats = zip(*data_stats)

        # Copy over comment.txt
        shutil.copyfile(os.path.join(self.data_directory, 'comment.txt'), os.path.join(new_dir, 'comment.txt'))

        with open(os.path.join(new_dir, 'statistics_analysis.txt'), 'a') as text_file:
            text_file.write('RESULTS SUMMARY\n')
            text_file.write('Mean amplitude (V): ' + str(numpy.mean(data_stats[0])) + '\n')
            text_file.write('Amplitude standard deviation (V): ' + str(numpy.std(data_stats[0])) + '\n\n')

            text_file.write('Mean max amplitude current (A): ' + str(numpy.mean(data_stats[1])) + '\n')
            text_file.write('Max amplitude current standard deviation (A): ' + str(numpy.std(data_stats[1])) + '\n\n')

            text_file.write('Mean min amplitude current (A): ' + str(numpy.mean(data_stats[2])) + '\n')
            text_file.write('Min amplitude current standard deviation (A): ' + str(numpy.std(data_stats[2])) + '\n\n')

            text_file.write('Mean peak distance (V): ' + str(numpy.mean(data_stats[3])) + '\n')
            text_file.write('Peak distance standard deviation (V): ' + str(numpy.std(data_stats[3])) + '\n\n')

            text_file.write('Mean peak separation (A): ' + str(numpy.mean(data_stats[4])) + '\n')
            text_file.write('Peak separation standard deviation (A): ' + str(numpy.std(data_stats[4])) + '\n\n')

            text_file.write('Mean peak separation (number of samples): ' + str(numpy.mean(data_stats[5])) + '\n')
            text_file.write('Peak separation standard deviation (number of samples): ' + str(numpy.std(data_stats[5])))

        # Make and save plot
        plt.figure(2)  # off screen figure
        plt.clf()
        plt.grid(True)
        plt.xlim([self.min_current, self.max_current])
        for line in self.data:
            line.graph_trial_data()
        plt.legend()
        plt.savefig(os.path.join(new_dir, 'plots.png'))

        # Compute average response
        # Make Graph and Save
        plt.figure(2)  # off screen figure
        plt.clf()
        plt.grid(True)
        plt.xlim([self.min_current, self.max_current])
        x, y = self.get_average_response(1000)
        plt.plot(x, y)
        plt.savefig(os.path.join(new_dir, 'average_plot.png'))

        # Save average response to file
        average_response = zip(x, y)
        with open(os.path.join(new_dir, 'average_signal_data.txt'), 'w') as data_file:
            for x, y in average_response:
                data_file.write('\t'+str(x)+'\t'+str(y)+'\n')
        return

    def get_average_response(self, resolution, epsilon=2.5e-4):
        cutoff_multiplier = abs((self.max_current - self.min_current) / float(resolution))
        points = [i*cutoff_multiplier+self.min_current for i in range(resolution)]

        voltage_values = defaultdict(list)
        for data in self.data:
            current_data, voltage_data = data.normalized_data
            for point in points:
                for index, current in enumerate(current_data):
                    if current >= point-epsilon and current <= point+epsilon:
                        voltage_values[point].append(voltage_data[index])

        average_signal = []
        for point, voltages in voltage_values.items():
            average_signal.append((point, numpy.average(voltages)))
        average_signal = sorted(average_signal, key=lambda x: x[0])

        return zip(*average_signal)


class Data():

    def __init__(self, sample, path):

        self.path = path
        self.sample_group = sample
        self.sample_num = path[len(self.sample_group.data_directory)+6:-4]

        self.raw_data = self.extract_data_lvm()

        self.raw_data_cutoffs = (0, -1)
        self.trimmed_data = self.trim_data()
        self.trimmed_data = self.linear_reg_model()

        self.max_val_loc = None
        self.min_val_loc = None

        self.normalize_lower_bound = 0
        self.normalize_upper_bound = len(self.trimmed_data[0])-1
        self.normalized_data = self.normalize_data()

    def __str__(self):
        return self.path

    def extract_data_lvm(self):
        """
        Read the data from the lvm file and return the current and voltage data
        NOTE: it is assumed that the current data is the first column and voltage is the second in the lvm file
        """
        raw_data = map(float, open(self.path).read().split())
        raw_current = [raw_data[i] for i in range(len(raw_data)) if not i % 2]
        raw_voltage = [raw_data[i] for i in range(len(raw_data)) if i % 2]
        return raw_current, raw_voltage

    def linear_reg_model(self):
        """
        Use a linear map to convert the current data from the DAC to the actual current used
        must NOT be done to the raw data, otherwise the trimming process won't work
        """
        current_data, voltage_data = self.trimmed_data
        m = (self.sample_group.max_current - self.sample_group.min_current) / 3.132982
        b = self.sample_group.min_current + 0.211337*(self.sample_group.max_current - self.sample_group.min_current)/3.132982
        return [data*m + b for data in current_data], voltage_data

    def trim_data(self):
        """
        Given a single trial's current data in the form [-0.21932, -0.21932, -0.21932, ...] and voltage data in the same form,
        trim the data such that it clips out uninteresting activity at the beginning and end.
        Returns the data as a 2-element tuple (current_data, voltage_data), both of which are trimmed and of the same length.
        """
        current_data, voltage_data = self.raw_data

        # Manually defined thresholds
        LOWER_LIMIT = -0.20
        UPPER_LIMIT = 2.85

        # Bounds to be modified during iteration
        lower_bound = None
        upper_bound = None

        # Iterate O(n) style to find the cutoff points
        for i in range(len(current_data)):
            if not lower_bound and abs(current_data[i] - LOWER_LIMIT) < 10**-3:
                lower_bound = int(i)
            if not upper_bound and abs(current_data[i] - UPPER_LIMIT) < 10**-3:
                upper_bound = int(i)

        self.raw_data_cutoffs = lower_bound, upper_bound
        return current_data[lower_bound:upper_bound], voltage_data[lower_bound:upper_bound]

    def normalize_data(self):
        """
        Search for the max and min of the data within the given bounds then
        take the center point and shift the data vertically so that the center is at 0
        """
        current_data, voltage_data = self.trimmed_data

        # Save the new bounds for later access
        lower_bound = self.normalize_lower_bound
        upper_bound = self.normalize_upper_bound

        # Find the min and max of the voltage data
        peak = max(voltage_data[lower_bound:upper_bound])
        trough = min(voltage_data[lower_bound:upper_bound])

        """
        NOTE: These next two steps assume that the peak of the signal comes BEFORE the trough
              if this is NOT the case then change the [0] 'max_val_loc line' to [-1]
              and change the [-1] in the 'min_val_loc' line (min) to [0]
        """
        # Find the FIRST occurrence of the maximum value, as it may not be unique
        self.max_val_loc = [i for i, j in enumerate(voltage_data[lower_bound:upper_bound]) if j == peak][0] + lower_bound
        # Find the LAST occurrence of the minimum value, as it may not be unique
        self.min_val_loc = [i for i, j in enumerate(voltage_data[lower_bound:upper_bound]) if j == trough][-1] + lower_bound
        center = (peak + trough) / 2.0

        return current_data, [v - center for v in voltage_data]

    def change_normalize_bounds(self, lower, upper):
        """
        Re-normalize data with the given bounds if the bounds are valid, otherwise do nothing and print an error
        """
        if upper < len(self.normalized_data[0]) and lower < upper:
            self.normalize_upper_bound = upper
            self.normalize_lower_bound = lower
            self.normalized_data = self.normalize_data()
        else:
            print "Error: Something is wrong with normalize bounds"

    def statistics_data(self, stat_file):
        """
        Returns a tuple of data about the run and appends it to the statistics file
        """
        # Compute Numbers
        current_data, voltage_data = self.normalized_data
        peak_distance_v = abs(voltage_data[self.max_val_loc] - voltage_data[self.min_val_loc])
        max_current = current_data[self.max_val_loc]
        min_current = current_data[self.min_val_loc]
        peak_distance_a = abs(max_current - min_current)
        peak_distance_s = abs(self.max_val_loc - self.min_val_loc)

        # Append to file
        with open(stat_file, 'a') as text_file:
            text_file.write('TRIAL ' + os.path.basename(self.path) + '\n')
            text_file.write('Amplitude (V): ' + str(peak_distance_v / 2.0) + '\n')
            text_file.write('Max amplitude current (A): ' + str(max_current) + '\n')
            text_file.write('Min amplitude current (A): ' + str(min_current) + '\n')
            text_file.write('Distance between peaks (V): ' + str(peak_distance_v) + '\n')
            text_file.write('Distance between peaks (A): ' + str(peak_distance_a) + '\n')
            text_file.write('Distance between peaks (number of samples): ' + str(peak_distance_s) + '\n\n')

        return peak_distance_v/2.0, max_current, min_current, peak_distance_v, peak_distance_a, peak_distance_s

    def graph_trial_data(self, cutoffs=False):
        x, y = self.normalized_data
        plt.plot(x, y, label='Trial '+self.sample_num)
        if cutoffs:
            plt.plot([x[self.normalize_lower_bound], x[self.normalize_upper_bound]], [y[self.max_val_loc], y[self.max_val_loc]], color='g', linestyle='--')
            plt.plot([x[self.normalize_lower_bound], x[self.normalize_upper_bound]], [y[self.min_val_loc], y[self.min_val_loc]], color='g', linestyle='--')
            plt.axvline(x=x[self.normalize_lower_bound], ymin=0, ymax=1, hold=None, color='r', linestyle='--')
            plt.axvline(x=x[self.normalize_upper_bound], ymin=0, ymax=1, hold=None, color='r', linestyle='--')


# Create Functions For buttons
def save():
    app.trial.save_data()
    return


def next_sample():
    app.trial.select_next_sample()
    app.sampleNum_label.config(text='Sample '+app.trial.current_sample().sample_num)
    return


def previous_sample():
    app.trial.select_previous_sample()
    app.sampleNum_label.config(text='Sample '+app.trial.current_sample().sample_num)
    return


def update_bounds():
    # Try to plot with new bounds
    lower_bound = int(app.lowerSearchBound_entry.get())
    upper_bound = int(app.upperSearchBound_entry.get())
    app.trial.current_sample().change_normalize_bounds(lower_bound, upper_bound)
    app.trial.plot_data()

    # Update text in entries with actual bounds used
    app.lowerSearchBound_entry.delete(0, END)
    app.upperSearchBound_entry.delete(0, END)
    app.lowerSearchBound_entry.insert(0, str(app.trial.current_sample().normalize_lower_bound))
    app.upperSearchBound_entry.insert(0, str(app.trial.current_sample().normalize_upper_bound))
    return


def change_directory():
    # Open file selection box
    selected_dir = tkFileDialog.askdirectory()
    app.trial = Trial(selected_dir)
    app.trial.plot_data()
    return


def open_trial():
    # Open file selection box
    selected_dir = tkFileDialog.askdirectory()
    app.trial = Trial(selected_dir)
    app.trial.plot_data()
    return


def create_new_trial():
    return


def multi_trial_stats():
    """
    Note: a lot of this should probably be in helper function tbh
    May want to add recursive copy of trial folders into summary folder, idk
    I also forgot to take sensitivity into account, I'll fix that once this all works
    """
    # Ask user for directory with all data directories to be used inside
    selected_dir = tkFileDialog.askdirectory()
    sub_dirs = [os.path.join(selected_dir, name) for name in os.listdir(selected_dir) if os.path.isdir(os.path.join(selected_dir, name))]

    # Create a TrialSummary Object for each trial
    trial_summaries = []
    for trial_dir in sub_dirs:
        trial_summaries.append(TrialSummary(trial_dir))

    # Create Plot of Averages and save
    plt.figure(2)  # off screen figure
    plt.clf()
    plt.grid(True)
    for summary in trial_summaries:
        x, y = summary.get_average_response()
        plt.plot(x, y, label=summary.name)
    plt.legend()
    plt.savefig(os.path.join(selected_dir, 'average_plot.png'))

    # Check if each trial can be given a number based on sample name
    summary_plot_numbered = []
    flag = True
    for trial in trial_summaries:
        try:
            num = float(trial.name)
            summary_plot_numbered.append((num, trial))
        except:
            flag = False
            break

    if flag:
        # Sort list by associated trial number (lest to greatest)
        summary_plot_numbered = sorted(summary_plot_numbered, key=lambda i: i[0])

    # Create the Summary Plots
    measurement_title = {0: "Amplitude (V)", 2: "Max amplitude current (A)", 4: "Min amplitude current (A)", 6: "Peak distance (V)", 8: "Peak separation (A)"}
    for measurement in range(0, 9, 2):
        plt.figure(2)
        plt.clf()
        values = [data.get_stats_list()[measurement] for number, data in summary_plot_numbered]
        errors = [data.get_stats_list()[measurement + 1] for number, data in summary_plot_numbered]
        plt.errorbar(range(len(values)), values, yerr=errors)
        for (label, x, y) in zip([x[0] for x in summary_plot_numbered], range(len(values)), values):
            plt.annotate(label, xy=(x, y), xytext=(5, 5), textcoords="offset points")
        plt.title(measurement_title[measurement])
        plt.xlabel("Sample")
        plt.ylabel("Value and error (one standard deviation)")
        plt.xlim([-1, len(values)])
        plt.savefig(os.path.join(selected_dir, measurement_title[measurement]))

    # Create Summary.txt
    for number, trial in summary_plot_numbered:
        trial.save_data(selected_dir)
    return


class Application(Frame):

    def __init__(self, master):

        # Create Trial (Defaults to parent directory of where the .py file is located)
        program_directory = os.getcwd()
        default_dir = os.path.dirname(program_directory)
        self.trial = Trial(default_dir)
        self.trial.plot_data()

        frame = Frame(master)
        frame.pack()

        # Create Main Frames
        self.leftFrame = Frame(master)
        self.leftFrame.pack(side=LEFT)

            # Create Trial Frames
        self.trialFrame = Frame(master)
        self.trialFrame.pack(side=RIGHT)

        self.topTrialFrame = Frame(self.trialFrame)
        self.topTrialFrame.pack(side=TOP)

        self.botTrialFrame = Frame(self.trialFrame)
        self.botTrialFrame.pack(side=BOTTOM)

            # Create New Trial Frames
        self.newTrialFrame = Frame(master)
        self.newTrialFrame.pack(side=RIGHT)

        self.botNewTrialFrame = Frame(self.newTrialFrame)
        self.botNewTrialFrame.pack(side=BOTTOM)

        self.topNewTrialFrame = Frame(self.newTrialFrame)
        self.topNewTrialFrame.pack(side=TOP)

        # Create Left Frame Widgets
        self.new_dir_btn = Button(self.leftFrame, text='Change Directory', command=change_directory)
        self.new_dir_btn.grid(row=0, column=0)

        self.new_trial_btn = Button(self.leftFrame, text='Create New Trial', command=create_new_trial)
        self.new_trial_btn.grid(row=1, column=0)

        self.open_trial_btn = Button(self.leftFrame, text='Open Trial', command=open_trial)
        self.open_trial_btn.grid(row=2, column=0)

        self.multi_trial_averages_btn = Button(self.leftFrame, text='Multi Trial Statistics', command=multi_trial_stats)
        self.multi_trial_averages_btn.grid(row=3, column=0)

        # Create Bottom Trial Frame Widgets
        self.save_btn = Button(self.botTrialFrame, text='Save', command=save)
        self.save_btn.grid(row=0, column=9)

        self.update_btn = Button(self.botTrialFrame, text='Update', command=update_bounds)
        self.update_btn.grid(row=0, column=8)

        self.nxtSample_btn = Button(self.botTrialFrame, text='Next Sample', command=next_sample)
        self.nxtSample_btn.grid(row=0, column=2)

        self.prvSample_btn = Button(self.botTrialFrame, text='Previous Sample', command=previous_sample)
        self.prvSample_btn.grid(row=0, column=0)

        self.sampleNum_label = Label(self.botTrialFrame, text='Sample 1')
        self.sampleNum_label.grid(row=0, column=1)

        self.searchWin_label = Label(self.botTrialFrame, text='to')
        self.searchWin_label.grid(row=0, column=6)

        self.lowerSearchBound_entry = Entry(self.botTrialFrame)
        self.lowerSearchBound_entry.grid(row=0, column=5)

        self.upperSearchBound_entry = Entry(self.botTrialFrame)
        self.upperSearchBound_entry.grid(row=0, column=7)

        # Create Top Trial Frame Canvases
        self.trial_graph = plt.figure(1)
        self.canvas = FigureCanvasTkAgg(self.trial_graph, master=self.topTrialFrame)
        plot_widget = self.canvas.get_tk_widget()
        plot_widget.pack()


# Create Main Frame
root = Tk()
root.title("EPR Data Analyzer")
app = Application(root)


def on_closing():
    """
    This function is needed for closing the tkinter window to also close out of the python interpreter
    """
    plt.close('all')
    root.destroy()

# Keep GUI Open
root.protocol('WM_DELETE_WINDOW', on_closing)
root.mainloop()