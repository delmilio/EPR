from Tkinter import *
import os
# from comp182 import *

import matplotlib
matplotlib.use('TkAgg')

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt


class Data():

    def __init__(self, file_dir, data_dir):

        self.path = os.path.join(data_dir, file_dir)
        self.raw_data = self.extract_data_lvm(self.path)
        self.trimmed_data = self.trim_data(self.raw_data)


    def extract_data_lvm(self, file_path):
        raw_data = map(float, open(file_path).read().split())
        raw_current = {i/2: data_point for (i, data_point) in enumerate(raw_data) if not i % 2}
        raw_voltage = {i/2: data_point for (i, data_point) in enumerate(raw_data) if i % 2}
        # print "Raw Data:    ", raw_data
        # print "Current Data:", raw_current
        # print "Voltage Data:", raw_voltage
        return raw_current, raw_voltage

    def trim_data(self, (current_data, voltage_data)):
        """
        Given a single trial's current data in the form [-0.21932, -0.21932, -0.21932, ...] and voltage data in the same form,
        trim the data such that it clips out uninteresting activity at the beginning and end.
        Returns the data as a 2-element tuple (current_data, voltage_data), both of which are trimmed and of the same length.
        """
        # Manually defined thresholds
        LOWER_LIMIT = -0.20
        UPPER_LIMIT = 2.85

        # Bounds to be modified during iteration
        lower_bound = None
        upper_bound = None

        # Convert Dictionaries to lists
        current_data = [value for (i, value) in sorted(current_data.items(), key=lambda x: x[0])]
        voltage_data = [value for (i, value) in sorted(voltage_data.items(), key=lambda x: x[0])]
        # print current_data
        # print voltage_data

        # Iterate O(n) style to find the cutoff points
        for i in range(len(current_data)):
            if not lower_bound and abs(current_data[i] - LOWER_LIMIT) < 10**-3:
                lower_bound = int(i)
            if not upper_bound and abs(current_data[i] - UPPER_LIMIT) < 10**-3:
                upper_bound = int(i)

        # trim data and place in dictionary
        current_data = {(i+lower_bound): value for (i, value) in enumerate(current_data[lower_bound:upper_bound])}
        voltage_data = {(i+lower_bound): value for (i, value) in enumerate(voltage_data[lower_bound:upper_bound])}

        return current_data, voltage_data

    def plot_sample(self):

        raw_current, raw_signal = self.raw_data
        trim_current, trim_signal = self.trimmed_data

        raw_current = [(i, value) for (i, value) in sorted(raw_current.items(), key=lambda x: x[0])]
        raw_current_x, raw_current_y = [i for (i, value) in raw_current], [value for (i, value) in raw_current]

        raw_signal = [(i, value) for (i, value) in sorted(raw_signal.items(), key=lambda x: x[0])]
        raw_signal_x, raw_signal_y = [i for (i, value) in raw_signal], [value for (i, value) in raw_signal]

        trim_current = [(i, value) for (i, value) in sorted(trim_current.items(), key=lambda x: x[0])]
        trim_current_x, trim_current_y = [i for (i, value) in trim_current], [value for (i, value) in trim_current]

        trim_signal = [(i, value) for (i, value) in sorted(trim_signal.items(), key=lambda x: x[0])]
        trim_signal_x, trim_signal_y = [i for (i, value) in trim_signal], [value for (i, value) in trim_signal]

        # Select first sub-plot and plot the raw data
        plt.subplot(211)
        plt.plot(raw_current_x, raw_current_y, raw_signal_x, raw_signal_y)

        # Select second sub-plot and plot trimmed data
        plt.subplot(212)
        plt.plot(trim_current_x, trim_current_y, trim_signal_x, trim_signal_y)

        # Update the graph
        # app.update_single_graph()
        return


class Application(Frame):

    def __init__(self, master):
        frame = Frame(master)
        frame.pack()

        # Create Frames
        self.leftFrame = Frame(master)
        self.leftFrame.pack(side=LEFT)
        self.rightFrame = Frame(master)
        self.rightFrame.pack(side=RIGHT)

        # Create Left Frame Buttons
        self.publish_btn = Button(self.leftFrame, text='     Publish     ')
        self.refactor_btn = Button(self.leftFrame, text='Refactor Data')
        self.refresh_btn = Button(self.leftFrame, text=' Refresh Data ')
        self.publish_btn.grid(columnspan=3)
        self.refactor_btn.grid(columnspan=3)
        self.refresh_btn.grid(columnspan=3)

        # Create Right Frame Canvases
        self.graph = plt.figure(1)
        self.canvas = FigureCanvasTkAgg(self.graph, master=self.rightFrame)
        plot_widget = self.canvas.get_tk_widget()
        plot_widget.pack()

        # Create vars
        self.data_entries = []
        self.radio_selection = IntVar()

        # Find Data Directory
        working_directory = os.getcwd()
        data_directory = working_directory[:working_directory.rfind('\\')]

        # Find the Data Files
        data_files = [data_file for data_file in os.listdir(data_directory) if data_file.find('data_') >= 0 and data_file.find('.lvm') >= 0]

        self.data_list = []
        # Display Data Files
        for data_file in data_files:
            self.data_list.append(Data(data_file, data_directory))
            self.add_data_label(data_file)

    def add_data_label(self, name):
        label = Label(self.leftFrame, text=name)

        cb_val = IntVar()
        cb_val.set(1)
        check_box = Checkbutton(self.leftFrame, variable=cb_val, onvalue=1, offvalue=0)

        r_button = Radiobutton(self.leftFrame, variable=self.radio_selection, value=len(self.data_entries))
        r_button.bind('<Button-1>', self.update_single_graph)

        self.data_entries.append((check_box, cb_val, label))
        label.grid(row=len(self.data_entries)+2, column=2)
        check_box.grid(row=len(self.data_entries)+2, column=1)
        r_button.grid(row=len(self.data_entries)+2, column=0)

    def update_single_graph(self, event):
        self.clear_single_graph()
        self.data_list[self.radio_selection.get()].plot_sample()
        self.graph.canvas.draw()

    def clear_single_graph(self):
        self.graph.clf()
        self.graph.canvas.draw()


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