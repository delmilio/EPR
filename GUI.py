from Tkinter import *
from collections import defaultdict
import os, datetime, time, numpy, shutil, tkFileDialog

import matplotlib
matplotlib.use('TkAgg')

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt


class Trial():

    def __init__(self, directory):

        # Find Data Directory
        self.data_directory = directory

        # Find Comment File
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
            line.graph_data()

        # Plot selected signal with bound lines in subplot 2
        plt.subplot(212)
        plt.grid(True)
        plt.xlim([self.min_current, self.max_current])
        self.current_sample().graph_data(True)

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

        with open(os.path.join(new_dir, 'statistics_analysis.txt'), 'a') as text_file:
            text_file.write('RESULTS SUMMARY\n')
            text_file.write('Mean amplitude (V): ' + str(numpy.mean(data_stats[0])) + '\n')
            text_file.write('Amplitude standard deviation (V): ' + str(numpy.std(data_stats[0])) + '\n\n')

            text_file.write('Mean max amplitude current (A): ' + str(numpy.mean(data_stats[1])) + '\n')
            text_file.write('Max amplutude current standard deviation (A): ' + str(numpy.std(data_stats[1])) + '\n\n')

            text_file.write('Mean min amplitude current (A): ' + str(numpy.mean(data_stats[2])) + '\n')
            text_file.write('Min amplutude current standard deviation (A): ' + str(numpy.std(data_stats[2])) + '\n\n')

            text_file.write('Mean peak distance (V): ' + str(numpy.mean(data_stats[3])) + '\n')
            text_file.write('Peak distance standard deviation (V): ' + str(numpy.std(data_stats[3])) + '\n\n')

            text_file.write('Mean peak separation (A): ' + str(numpy.mean(data_stats[4])) + '\n')
            text_file.write('Peak separation standard deviation (A): ' + str(numpy.std(data_stats[4])) + '\n\n')

            text_file.write('Mean peak separation (number of samples): ' + str(numpy.mean(data_stats[5])) + '\n')
            text_file.write('Peak separation standard deviation (number of samples): ' + str(numpy.std(data_stats[5])))

        # Make and save plot
        plt.figure(2)
        plt.clf()
        plt.grid(True)
        plt.xlim([self.min_current, self.max_current])
        for line in self.data:
            line.graph_data()
        plt.legend()
        plt.savefig(os.path.join(new_dir, 'plots.png'))

        # Compute average response
        # Make Graph and Save
        plt.figure(3)
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
        # Find the FIRST occurence of the maximum value, as it may not be unique
        self.max_val_loc = [i for i, j in enumerate(voltage_data[lower_bound:upper_bound]) if j == peak][0] + lower_bound
        # Find the LAST occurence of the minimum value, as it may not be unique
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
            text_file.write('TRIAL ' + str(self.path)[22:] + '\n')
            text_file.write('Amplitude (V): ' + str(peak_distance_v / 2.0) + '\n')
            text_file.write('Max amplitude current (A): ' + str(max_current) + '\n')
            text_file.write('Min amplitude current (A): ' + str(min_current) + '\n')
            text_file.write('Distance between peaks (V): ' + str(peak_distance_v) + '\n')
            text_file.write('Distance between peaks (A): ' + str(peak_distance_a) + '\n')
            text_file.write('Distance between peaks (number of samples): ' + str(peak_distance_s) + '\n\n')

        return peak_distance_v/2.0, max_current, min_current, peak_distance_v, peak_distance_a, peak_distance_s

    def graph_data(self, cutoffs=False):
        x, y = self.normalized_data
        plt.plot(x, y, label='Trial '+self.sample_num)
        if cutoffs:
            plt.plot([x[self.normalize_lower_bound], x[self.normalize_upper_bound]], [y[self.max_val_loc], y[self.max_val_loc]], color='g', linestyle='--')
            plt.plot([x[self.normalize_lower_bound], x[self.normalize_upper_bound]], [y[self.min_val_loc], y[self.min_val_loc]], color='g', linestyle='--')
            plt.axvline(x=x[self.normalize_lower_bound], ymin=0, ymax=1, hold=None, color='r', linestyle='--')
            plt.axvline(x=x[self.normalize_upper_bound], ymin=0, ymax=1, hold=None, color='r', linestyle='--')

program_directory = os.getcwd()
default_dir = os.path.dirname(program_directory)
trial = Trial(default_dir)
trial.plot_data()


# Create Functions For buttons
def save():
    trial.save_data()
    return


def next_sample():
    trial.select_next_sample()
    app.sampleNum_label.config(text='Sample '+trial.current_sample().sample_num)
    return


def previous_sample():
    trial.select_previous_sample()
    app.sampleNum_label.config(text='Sample '+trial.current_sample().sample_num)
    return


def update_bounds():
    # Try to plot with new bounds
    lower_bound = int(app.lowerSearchBound_entry.get())
    upper_bound = int(app.upperSearchBound_entry.get())
    trial.current_sample().change_normalize_bounds(lower_bound, upper_bound)
    trial.plot_data()

    # Update text in entries with actual bounds used
    app.lowerSearchBound_entry.delete(0, END)
    app.upperSearchBound_entry.delete(0, END)
    app.lowerSearchBound_entry.insert(0, str(trial.current_sample().normalize_lower_bound))
    app.upperSearchBound_entry.insert(0, str(trial.current_sample().normalize_upper_bound))
    return


def change_directory():
    # Open file selection box
    print tkFileDialog.askdirectory()
    return


class Application(Frame):

    def __init__(self, master):
        frame = Frame(master)
        frame.pack()

        # Create Main Frames
        self.leftFrame = Frame(master)
        self.leftFrame.pack(side=LEFT)

        self.rightFrame = Frame(master)
        self.rightFrame.pack(side=RIGHT)

        self.topRightFrame = Frame(self.rightFrame)
        self.topRightFrame.pack(side=TOP)

        self.botRightFrame = Frame(self.rightFrame)
        self.botRightFrame.pack(side=BOTTOM)

        # Create Left Frame Widgets
        self.new_dir_btn = Button(self.leftFrame, text='Change Directory', command=change_directory)
        self.new_dir_btn.grid(row=0, column=0)

        self.new_trial_btn = Button(self.leftFrame, text='Create New Trial')
        self.new_trial_btn.grid(row=1, column=0)

        self.open_trial_btn = Button(self.leftFrame, text='Open Trial')
        self.open_trial_btn.grid(row=2, column=0)

        self.multi_trial_averages_btn = Button(self.leftFrame, text='Multipul Averages')
        self.multi_trial_averages_btn.grid(row=3, column=0)

        # Create Right Bottom Frame Widgets
        self.save_btn = Button(self.botRightFrame, text='Save', command=save)
        self.save_btn.grid(row=0, column=9)

        self.update_btn = Button(self.botRightFrame, text='Update', command=update_bounds)
        # self.update_btn.bind('<Return>', update_bounds)
        self.update_btn.grid(row=0, column=8)

        self.nxtSample_btn = Button(self.botRightFrame, text='Next Sample', command=next_sample)
        self.nxtSample_btn.grid(row=0, column=2)

        self.prvSample_btn = Button(self.botRightFrame, text='Previous Sample', command=previous_sample)
        self.prvSample_btn.grid(row=0, column=0)

        self.sampleNum_label = Label(self.botRightFrame, text='Sample 1')
        self.sampleNum_label.grid(row=0, column=1)

        self.searchWin_label = Label(self.botRightFrame, text='to')
        self.searchWin_label.grid(row=0, column=6)

        self.lowerSearchBound_entry = Entry(self.botRightFrame)
        self.lowerSearchBound_entry.grid(row=0, column=5)

        self.upperSearchBound_entry = Entry(self.botRightFrame)
        self.upperSearchBound_entry.grid(row=0, column=7)

        # Create Right Top Frame Canvases
        self.graph = plt.figure(1)
        self.canvas = FigureCanvasTkAgg(self.graph, master=self.topRightFrame)
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