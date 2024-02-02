import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import filedialog
from tkinter.ttk import Treeview
# import QuanTest1
# import QuanTest2
import math

import CompareSignal
import IDFTtest
import signalcompare
import comparesignal2
import Shift_Fold_Signal
from TestCases.Derivative import DerivativeSignal
from TestCases.Convolution import ConvTest


def read_signal_from_file(file_path):
    # Read signal samples from a text file
    with open(file_path) as file:
        listOfLines = file.readlines()
        IsPeriodic = listOfLines[0]
        SignalType = listOfLines[1]
        SignalType = int(SignalType.strip())
        sampleSize = listOfLines[2]
        del listOfLines[0]
        del listOfLines[0]
        del listOfLines[0]
        samplesX = []
        samplesY = []
        for line in listOfLines:
            if "," in line:
                if "f" in line:
                    splitList = line.strip().split(",")
                    splitList[0] = splitList[0].replace("f", "")
                    splitList[1] = splitList[1].replace("f", "")
                else:
                    splitList = line.strip().split(",")
            else:
                if "f" in line:
                    splitList = line.strip().split()
                    splitList[0] = splitList[0].replace("f", "")
                    splitList[1] = splitList[1].replace("f", "")
                else:
                    splitList = line.strip().split()
            sampleX = float(splitList[0])
            sampleY = float(splitList[1])
            samplesX.append(sampleX)
            samplesY.append(sampleY)
    return samplesX, samplesY, SignalType


def plot_signal_from_file(samplesX, samplesY):
    two_subplot_fig = plt.figure(figsize=(6, 6))
    plt.subplot(211)
    plt.plot(samplesX, samplesY, color="orange")
    plt.title("Continuous Signal")
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.subplot(212)
    plt.stem(samplesX, samplesY)
    plt.title("Discrete Signal")
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.subplots_adjust(0.1, 0.1, 0.9, 0.9, 0.4, 0.4)
    two_subplot_fig.show()


def generate_sinusoidal_signal(amplitude, phase_shift, analog_frequency, sampling_frequency, duration):
    # Generate a sinusoidal signal
    if sampling_frequency != 0:
        t = np.arange(0, duration, 1 / sampling_frequency)
    else:
        t = np.arange(0, duration, 1 / analog_frequency)
    signal = amplitude * np.sin(2 * np.pi * analog_frequency * t + phase_shift)
    return signal


def generate_cosinusoidal_signal(amplitude, phase_shift, analog_frequency, sampling_frequency, duration):
    # Generate a cosinusoidal signal
    if sampling_frequency != 0:
        t = np.arange(0, duration, 1 / sampling_frequency)
    else:
        t = np.arange(0, duration, 1 / analog_frequency)
    signal = amplitude * np.cos(2 * np.pi * analog_frequency * t + phase_shift)
    return signal


def plot_signal_wave(samples, title, flag1, flag2=""):
    two_subplot_fig = plt.figure(figsize=(6, 6))
    if flag1 == "error" or flag2 == "error":
        plt.subplot(211)
        plt.plot(samples, color="orange")
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.subplot(212)
        plt.stem([0, 0, 0, 0, 0, 0, 0, 0])
        plt.title(title)
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.subplots_adjust(0.1, 0.1, 0.9, 0.9, 0.4, 0.4)
        two_subplot_fig.show()
    else:
        plt.subplot(211)
        plt.plot(samples, color="orange")
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.subplot(212)
        plt.stem(samples)
        plt.title(title)
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.subplots_adjust(0.1, 0.1, 0.9, 0.9, 0.4, 0.4)
        two_subplot_fig.show()


def generate_wave_from_file():
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    x, y, signal_type = read_signal_from_file(file_path)
    plot_signal_from_file(x, y)


def signal_generation_menu_gui():
    # Create a new window for the signal generation menu
    generation_window = Toplevel(root)
    generation_window.title("Signal Generation Menu")
    generation_window.geometry("300x250")

    signal_type_label = Label(generation_window, text="Signal Type(sin || cos):")
    signal_type_label.grid(row=0, column=0)
    signal_type_entry = Entry(generation_window)
    signal_type_entry.grid(row=0, column=1)

    amplitude_label = Label(generation_window, text="Amplitude:")
    amplitude_label.grid(row=1, column=0)
    amplitude_entry = Entry(generation_window)
    amplitude_entry.grid(row=1, column=1)

    phase_shift_label = Label(generation_window, text="Phase Shift (radians):")
    phase_shift_label.grid(row=2, column=0)
    phase_shift_entry = Entry(generation_window)
    phase_shift_entry.grid(row=2, column=1)

    analog_frequency_label = Label(generation_window, text="Analog Frequency (Hz):")
    analog_frequency_label.grid(row=3, column=0)
    analog_frequency_entry = Entry(generation_window)
    analog_frequency_entry.grid(row=3, column=1)

    sampling_frequency_label = Label(generation_window, text="Sampling Frequency (Hz):")
    sampling_frequency_label.grid(row=4, column=0)
    sampling_frequency_entry = Entry(generation_window)
    sampling_frequency_entry.grid(row=4, column=1)

    duration_label = Label(generation_window, text="Duration (seconds):")
    duration_label.grid(row=5, column=0)
    duration_entry = Entry(generation_window)
    duration_entry.grid(row=5, column=1)

    def generate_wave():
        signal_type = signal_type_entry.get()
        amplitude = float(amplitude_entry.get())
        phase_shift = float(phase_shift_entry.get())
        analog_frequency = float(analog_frequency_entry.get())
        sampling_frequency = float(sampling_frequency_entry.get())
        duration = float(duration_entry.get())
        flag2 = ""
        if sampling_frequency == 0:
            flag1 = "error"
        else:
            flag1 = "ok"
            if analog_frequency / sampling_frequency > 0.5 or analog_frequency / sampling_frequency < -0.5:
                flag2 = "error"
            else:
                flag2 = "ok"

        if signal_type == "sin":
            signal1 = generate_sinusoidal_signal(amplitude, phase_shift, analog_frequency, sampling_frequency, duration)
            title = f"Sine Wave: A={amplitude}, θ={phase_shift}, f={analog_frequency}, fs={sampling_frequency}"
            plot_signal_wave(signal1, title, flag1, flag2)
        else:
            signal2 = generate_cosinusoidal_signal(amplitude, phase_shift, analog_frequency, sampling_frequency,
                                                   duration)
            title = f"Cosine Wave: A={amplitude}, θ={phase_shift}, f={analog_frequency}, fs={sampling_frequency}"
            plot_signal_wave(signal2, title, flag1, flag2)

    generate_signal_button = Button(generation_window, text="Generate Signal Wave", bg="black", fg="white",
                                    command=generate_wave)
    generate_signal_button.grid(row=6, column=1)


def add_signals_gui():
    sig_num_window = Toplevel(root)
    sig_num_window.title("Enter Number Of Signals")

    signal_num_label = Label(sig_num_window, text="Number of signals:")
    signal_num_label.grid(row=0, column=0)
    global signal_num_entry
    signal_num_entry = Entry(sig_num_window)
    signal_num_entry.grid(row=0, column=1)

    enter_button = Button(sig_num_window, text="Enter", command=add_waves)
    enter_button.grid(row=1, column=1)


def add_waves():
    num_of_signals = int(signal_num_entry.get())
    input_y = []
    output_y = []
    for _ in range(num_of_signals):
        file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        x, y, signal_type = read_signal_from_file(file_path)
        input_y.append(y)
    output_y = [sum(signal_values) for signal_values in zip(*input_y)]

    plt.figure()
    plt.plot(x, output_y)
    plt.title("Output Signal")
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.show()


def sub_signals_gui():
    output_y = []

    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    x1, y1, signal_type = read_signal_from_file(file_path)
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    x2, y2, signal_type = read_signal_from_file(file_path)
    output_y = [amp1 - amp2 for amp1, amp2 in zip(y1, y2)]

    fig, axs = plt.subplots(3)
    axs[0].plot(x1, y1)
    axs[0].set_title("Signal1")
    axs[0].set(xlabel='Sample', ylabel='Amplitude')

    axs[1].plot(x2, y2)
    axs[1].set_title("Signal2")
    axs[1].set(xlabel='Sample', ylabel='Amplitude')

    axs[2].plot(x1, output_y)
    axs[2].set_title("Output Signal")
    axs[2].set(xlabel='Sample', ylabel='Amplitude')

    fig.tight_layout()

    fig.show()


def multiply_by_constant_gui():
    constant_window = Toplevel(root)
    constant_window.title("Enter Your Constant")

    constant_label = Label(constant_window, text="Constant:")
    constant_label.grid(row=0, column=0)
    global constant_entry
    constant_entry = Entry(constant_window)
    constant_entry.grid(row=0, column=1)

    enter_button = Button(constant_window, text="Enter", command=multiply_signal)
    enter_button.grid(row=1, column=1)


def multiply_signal():
    output_y = []
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    x, y, signal_type = read_signal_from_file(file_path)

    const = float(constant_entry.get())
    for i in y:
        output_y.append(i * const)
    fig, axs = plt.subplots(2)

    axs[0].plot(x, y)
    axs[0].set_title("Original Signal")
    axs[0].set(xlabel='Sample', ylabel='Amplitude')

    axs[1].plot(x, output_y)
    axs[1].set_title("Edited Signal")
    axs[1].set(xlabel='Sample', ylabel='Amplitude')

    fig.tight_layout()

    fig.show()


def square_signal_gui():
    output_y = []
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    x, y, signal_type = read_signal_from_file(file_path)
    for i in y:
        output_y.append(i * i)
    fig, axs = plt.subplots(2)

    axs[0].plot(x, y)
    axs[0].set_title("Original Signal")
    axs[0].set(xlabel='Sample', ylabel='Amplitude')

    axs[1].plot(x, output_y)
    axs[1].set_title("Squared Signal")
    axs[1].set(xlabel='Sample', ylabel='Amplitude')

    fig.tight_layout()

    fig.show()


def shift_signal_gui():
    constant_window = Toplevel(root)
    constant_window.title("Enter Your Constant")

    constant_label = Label(constant_window, text="Constant:")
    constant_label.grid(row=0, column=0)
    global constant_entry
    constant_entry = Entry(constant_window)
    constant_entry.grid(row=0, column=1)

    enter_button = Button(constant_window, text="Enter", command=shift_signal)
    enter_button.grid(row=1, column=1)


def shift_signal():
    output_x = []
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    x, y, signal_type = read_signal_from_file(file_path)

    const = float(constant_entry.get())
    for i in x:
        output_x.append(i + const)
    fig, axs = plt.subplots(2)

    axs[0].plot(x, y)
    axs[0].set_title("Original Signal")
    axs[0].set(xlabel='Sample', ylabel='Amplitude')

    axs[1].plot(output_x, y)
    axs[1].set_title("Shifted Signal")
    axs[1].set(xlabel='Sample', ylabel='Amplitude')

    fig.tight_layout()

    fig.show()


def normalize_signal_gui():
    normalize_type_window = Toplevel(root)
    normalize_type_window.title("Enter Valid Normalize Type")

    choice_label = Label(normalize_type_window, text="Type (-1 to 1 || 0 to 1):")
    choice_label.grid(row=0, column=0)
    global choice_entry
    choice_entry = Entry(normalize_type_window)
    choice_entry.grid(row=0, column=1)

    enter_button = Button(normalize_type_window, text="Enter", command=normalize_signal)
    enter_button.grid(row=1, column=1)


def normalize_signal():
    output_y = []
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    x, y, signal_type = read_signal_from_file(file_path)
    min_value = min(y)
    max_value = max(y)

    normalize_type = choice_entry.get()
    for i in y:
        if normalize_type == "-1 to 1":
            normalized_signal = 2 * ((i - min_value) / (max_value - min_value)) - 1
            output_y.append(normalized_signal)
        elif normalize_type == "0 to 1":
            normalized_signal = (i - min_value) / (max_value - min_value)
            output_y.append(normalized_signal)
        else:
            raise ValueError("Invalid normalize_type.\nEnter -1 to 1 or 0 to 1")
    fig, axs = plt.subplots(2)

    axs[0].plot(x, y)
    axs[0].set_title("Original Signal")
    axs[0].set(xlabel='Sample', ylabel='Amplitude')

    axs[1].plot(x, output_y)
    axs[1].set_title("Normalized Signal")
    axs[1].set(xlabel='Sample', ylabel='Amplitude')

    fig.tight_layout()

    fig.show()


def accumulative_sum_gui():
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    x, y, signal_type = read_signal_from_file(file_path)
    count = 0
    output_y = []
    for i in y:
        count += i
        output_y.append(count)
    fig, axs = plt.subplots(2)

    axs[0].plot(x, y)
    axs[0].set_title("Original Signal")
    axs[0].set(xlabel='Sample', ylabel='Amplitude')

    axs[1].plot(x, output_y)
    axs[1].set_title("Accumulated Signal")
    axs[1].set(xlabel='Sample', ylabel='Amplitude')

    fig.tight_layout()

    fig.show()


def quantize_signal_gui():
    def num_of_bits_fun():
        num_of_bits_window = Toplevel(quantize_window)
        num_of_bits_window.title("Enter Number Of Bits")

        num_of_bits_label = Label(num_of_bits_window, text="Number Of Bits:")
        num_of_bits_label.grid(row=0, column=0)
        global num_of_bits_entry
        num_of_bits_entry = Entry(num_of_bits_window)
        num_of_bits_entry.grid(row=0, column=1)

        enter_button = Button(num_of_bits_window, text="Enter", command=quantize_signal)
        enter_button.grid(row=1, column=1)

    def num_of_levels_fun():
        num_of_levels_window = Toplevel(root)
        num_of_levels_window.title("Enter Number Of Levels")

        num_of_levels_label = Label(num_of_levels_window, text="Number Of Levels:")
        num_of_levels_label.grid(row=0, column=0)
        global num_of_levels_entry
        num_of_levels_entry = Entry(num_of_levels_window)
        num_of_levels_entry.grid(row=0, column=1)

        enter_button = Button(num_of_levels_window, text="Enter", command=quantize_signal)
        enter_button.grid(row=1, column=1)

    quantize_window = Toplevel(root)
    quantize_window.title("Your Choice?")

    radio_button = IntVar()
    r1 = Radiobutton(quantize_window, text="Number Of Levels", variable=radio_button, value=1,
                     command=num_of_levels_fun)
    r1.pack(anchor=W)

    r2 = Radiobutton(quantize_window, text="Number Of Bits", variable=radio_button, value=2, command=num_of_bits_fun)
    r2.pack(anchor=W)

    def quantize_signal():
        ######### declarations ######

        file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        x, y, signal_type = read_signal_from_file(file_path)
        min_y = min(y)
        max_y = max(y)
        intervals = []
        midpoints = []
        indices = []
        quantized_error = []
        encoded_values = []

        ######### calc num_of_levels ######

        if radio_button.get() == 2:
            num_of_bits = int(num_of_bits_entry.get())
            num_of_levels = 2 ** num_of_bits
        else:
            num_of_levels = int(num_of_levels_entry.get())
            num_of_bits = round(math.log2(num_of_levels))

        # ~~~ calc delta, intervals and midpoints ~~~
        delta = (max_y - min_y) / num_of_levels
        new_max = min_y + delta
        while min_y < max_y:
            intervals.append([min_y, new_max])
            min_y = round((min_y + delta), 3)
            new_max = round((min_y + delta), 3)

        for i in intervals:
            midpoints.append(round((i[0] + i[1]) / 2, 3))

        quantized_values = []

        def find_interval(inpt):
            count = 0
            num_of_levels_of_var = 0
            for k in range(len(intervals)):
                min_val = intervals[k][0]
                max_val = intervals[k][1]
                if min_val <= inpt:
                    if inpt <= max_val:
                        num_of_levels_of_var = count
                        indices.append(num_of_levels_of_var + 1)
                        quantized_values.append(midpoints[num_of_levels_of_var])
                        encoded_values.append(format(num_of_levels_of_var, f'0{num_of_bits}b'))
                        break
                count = count + 1
            return num_of_levels_of_var

        # ~~~~ fitch the interval of the input ~~~~~~~
        for inpt in y:
            interval_number = find_interval(inpt)
            quantized_error.append(round(midpoints[interval_number] - inpt, 3))
        squared_error = []
        for m in quantized_error:
            squared_error.append(round(m * m, 3))
        # squared_error = squared_error / len(quantized_error)

        squared_error = []
        for m in quantized_error:
            squared_error.append(round(m * m, 3))

        # Create the table
        table_window = Toplevel(quantize_window)
        table_window.title("Table")

        table = Treeview(table_window)
        table["columns"] = ("X(n)", "Interval Index", "Q(n)", "Error", "E^2", "Encoded Values")
        table.heading("X(n)", text="X(n)")
        table.heading("Interval Index", text="Interval Index")
        table.heading("Q(n)", text="Q(n)")
        table.heading("Error", text="Error")
        table.heading("E^2", text="E^2")
        table.heading("Encoded Values", text="Encoded Values")

        # Populate the table
        for i in range(len(y)):
            n = y[i]
            interval_idx = indices[i]
            q = quantized_values[i]
            error = quantized_error[i]
            sq_error = squared_error[i]
            enc_values = encoded_values[i]
            table.insert("", "end", values=(n, interval_idx, q, error, sq_error, enc_values))

        table.pack()

        # if radio_button.get() == 2:
        # QuanTest1.QuantizationTest1('Quan1_Out.txt', encoded_values, quantized_values)
        # elif radio_button.get() == 1:
        # QuanTest2.QuantizationTest2('Quan2_Out.txt', indices, encoded_values, quantized_values, quantized_error)


# ~~~~~~ Task 4 ~~~~~~~~

def fourier_transform_gui():
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    x, y, signal_type = read_signal_from_file(file_path)

    def fourier_transform():
        if signal_type == 0:
            dft()
        # ~~~~~ IDFT compare signal ~~~~~~
        else:
            idft()
            file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
            correct_y = IDFTtest.output_fun(file_path)
            IDFTtest.my_output_fun(x_of_n, correct_y)

    # if work with DFT create the text_input to take the sampling frequency ############################################
    if signal_type == 0:
        sampling_frequency_window = Toplevel(root)
        sampling_frequency_window.title("Enter Sampling Frequency in HZ")

        sampling_frequency_label = Label(sampling_frequency_window, text="Sampling Frequency:")
        sampling_frequency_label.grid(row=0, column=0)

        global sampling_frequency_entry
        sampling_frequency_entry = Entry(sampling_frequency_window)
        sampling_frequency_entry.grid(row=0, column=1)

        enter_button = Button(sampling_frequency_window, text="Enter")
        enter_button.grid(row=1, column=1)

        enter_button = Button(sampling_frequency_window, text="Enter", command=fourier_transform)
        enter_button.grid(row=1, column=1)
    else:
        fourier_transform()


# ~~~~~~ DFT ~~~~~~~
def dft(x_Of_k = None, sampling_freq = None):
    x_of_k = []
    if x_Of_k is None:
        file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        x, y, signal_type = read_signal_from_file(file_path)
        global amplituedes
        global phase_shifts
        amplituedes = []
        phase_shifts = []
        for k in range(len(x)):
            count = 0
            for n in range(len(x)):
                # 1st harmonic x(0) ######################################
                if n * k == 0:
                    count = count + y[n]
                    count = complex(count)
                # other n harmonics ######################################
                else:
                    degree = (2 * math.pi * n * k) / len(x)
                    cos = round(math.cos(degree), 15)
                    sin = round(math.sin(degree), 15)
                    exp = complex(cos, -sin)
                    count = count + y[n] * exp
            # append x(k) in the list of harmonics ########################
            x_of_k.append(count)
    else:
        x_of_k = x_Of_k
        amplituedes = []
        phase_shifts = []

    # compute frequencies ,amplitudes and phase shifts #########################################################
    if sampling_freq == None:
        sampling_freq = float(sampling_frequency_entry.get())
    fundamental_freq = (2 * np.pi * sampling_freq) / len(x_of_k)
    frequencies = []
    for a in range(len(x_of_k)):
        # Amplitude ######################################
        amp = math.sqrt(math.pow(x_of_k[a].real, 2) + math.pow(x_of_k[a].imag, 2))
        amplituedes.append(amp)
        # Phase Shift ####################################
        phase = np.arctan2(x_of_k[a].imag, x_of_k[a].real)
        phase_shifts.append(phase)
        # Frequency ######################################
        frequencies.append(fundamental_freq * (a + 1))

    # ~~~ save Amplitude and Phase in text file ~~~~~~
    with open('my_output.txt', 'w') as f:
        f.write("0")
        f.write("\n")
        f.write("1")
        f.write("\n")
        f.write(str(len(phase_shifts)))
        for m, n in zip(amplituedes, phase_shifts):
            m = str(m)
            n = str(n)
            f.write("\n")
            f.write(m)
            f.write(" ")
            f.write(n)
    # ~~~ plot "Frequency vs Amplitude" & "Frequency vs Phase Shift" ~~~
    fig, axs = plt.subplots(2)
    axs[0].stem(frequencies, amplituedes)
    axs[0].set_title("Frequencies vs Amplitudes")
    axs[0].set(xlabel='Frequency', ylabel='Amplitude')

    axs[1].stem(frequencies, phase_shifts)
    axs[1].set_title("Frequencies vs Phase Shifts")
    axs[1].set(xlabel='Frequency', ylabel='Phase Shift')

    fig.tight_layout()

    fig.show()

    # ~~~~~ Modification function ~~~~~~
    def modify_file():
        index = int(modification_index_entry.get())
        amplitude = modification_Amplitude_value_entry.get()
        phase_shift = modification_Phase_value_entry.get()
        with open('my_output.txt', 'r') as file:
            lines = file.readlines()
        if 0 <= index < len(lines):

            lines[index + 3] = f"{amplitude} {phase_shift}\n"

            with open('my_output.txt', 'w') as file:
                file.writelines(lines)
        else:
            print(f"Invalid index: {index}")

    # ~~~~~ Modification Window ~~~~~~
    modification_window = Toplevel(root)
    modification_window.title("DFT Choice")
    modification_window.geometry("300x200")

    modification_label = Label(modification_window, text="Modification Index")
    modification_label.grid(row=0, column=0)
    global modification_index_entry
    modification_index_entry = Entry(modification_window)
    modification_index_entry.grid(row=0, column=1)

    amplitude_label = Label(modification_window, text="Amplitude:")
    amplitude_label.grid(row=1, column=0)
    global modification_Amplitude_value_entry
    modification_Amplitude_value_entry = Entry(modification_window)
    modification_Amplitude_value_entry.grid(row=1, column=1)

    phase_label = Label(modification_window, text="Phase Shift:")
    phase_label.grid(row=2, column=0)
    global modification_Phase_value_entry
    modification_Phase_value_entry = Entry(modification_window)
    modification_Phase_value_entry.grid(row=2, column=1)

    modify_button = Button(modification_window, text="Modify", command=modify_file)
    modify_button.grid(row=3, column=1)

    compare_button = Button(modification_window, text="Compare", command=compare_dft)
    compare_button.grid(row=4, column=1)
    return x_of_k, 'my_output.txt'


# ~~~~~ DFT compare signal ~~~~~~~

def compare_dft():
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    correct_amp, correct_phase, v_type = read_signal_from_file(file_path)
    bool1 = signalcompare.SignalComapreAmplitude(amplituedes, correct_amp)
    bool2 = signalcompare.SignalComaprePhaseShift(phase_shifts, correct_phase)
    print(bool1, bool2)


# ~~~~~~ IDFT ~~~~~~~~
def idft(file_path=None):
    if file_path is None:
        File_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    else:
        File_path = file_path
    if File_path is str:
        x, y, signal_type = read_signal_from_file(File_path)
    x_of_k = []
    global x_of_n
    x_of_n = []
    complex_numbers = []
    # convert the Amplitude and Phase Shift to the complex number form "real and imaginary"#####################
    for i in range(len(x)):
        # real = Amplitude * cos(phaseShift) ##########################
        real_number = x[i] * math.cos(y[i])
        # imaginary = Amplitude * sin(phaseShift) #####################
        imaginary_number = x[i] * math.sin(y[i])
        complex_number = complex(real_number, imaginary_number)
        complex_numbers.append(complex_number)
    # x(k) are complex numbers ########################################
    for n in range(len(x)):
        count = 0
        for k in range(len(x)):
            # 1st harmonic x(0) ######################################
            if n * k == 0:
                count = count + complex_numbers[k]

            # other n harmonics ######################################
            else:
                degree = (2 * math.pi * n * k) / len(x)
                cos = round(math.cos(degree), 12)
                sin = round(math.sin(degree), 12)
                exp = complex(cos, sin)
                count = count + (complex_numbers[k] * exp)

        count_real = round(count.real) / len(x)
        count_imagine = round(count.imag, 13)
        x_of_n.append(count_real)
        x_of_k.append(complex(count_real, count_imagine))
    print(x_of_k)
    print(x_of_n)

    time = range(len(x_of_n))
    plt.plot(time, x_of_n)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title("Signal plot in time")
    plt.grid(True)
    plt.show()
    return x_of_n


def idft3(x_Of_n):
    x_of_k = []
    global x_of_n
    x_of_n = []
    complex_numbers = []
    for n in range(len(x_Of_n)):
        count = 0
        for k in range(len(x_Of_n)):
            # 1st harmonic x(0) ######################################
            if n * k == 0:
                count = count + x_Of_n[k]

            # other n harmonics ######################################
            else:
                degree = (2 * math.pi * n * k) / len(x_Of_n)
                cos = round(math.cos(degree), 12)
                sin = round(math.sin(degree), 12)
                exp = complex(cos, sin)
                count = count + (x_Of_n[k] * exp)

        count_real = count.real / len(x_Of_n)
        count_imagine = round(count.imag, 11)
        x_of_n.append(count_real)
        x_of_k.append(complex(count_real, count_imagine))

    return x_of_n

# ~~~~~~~~~~~~~~~~~ Task 5 ~~~~~~~~~~~~~~~~~~~~~


def DCT_gui():
    dct_window = Toplevel(root)
    dct_window.title("DC Choice")
    dct_window.geometry("300x200")
    number_of_coefficients_label = Label(dct_window, text="Number of coefficients")
    number_of_coefficients_label.grid(row=0, column=0)
    global number_of_coefficients_entry
    number_of_coefficients_entry = Entry(dct_window)
    number_of_coefficients_entry.grid(row=0, column=1)
    compute_dct_button = Button(dct_window, text="Compute DCT", command=compute_dct)
    compute_dct_button.grid(row=1, column=1)

    remove_dct_button = Button(dct_window, text="Remove DCT", command=remove_dct)
    remove_dct_button.grid(row=2, column=1)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def compute_dct():
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    x, y, signal_type = read_signal_from_file(file_path)
    outer_term = math.sqrt(2 / len(x))
    my_output = []
    for k in range(len(x)):
        inner_term = 0
        for n in range(len(x)):
            angle = (np.pi / (4 * len(x))) * (2 * n - 1) * (2 * k - 1)
            cos_term = math.cos(angle)
            inner_term = inner_term + (y[n] * cos_term)

        my_output.append(outer_term * inner_term)
    print(my_output)
    with open('my_output2.txt', 'w') as file:

        file.write("0\n")
        file.write("1\n")
        file.write(f"{len(x)}\n")
        num_coefficients = int(number_of_coefficients_entry.get())
        if 0 < num_coefficients < len(x):
            for k in range(num_coefficients):
                file.write(f"{k} {my_output[k]:.4f}\n")
        else:
            print("number of coefficients out of range")
    expected_output_file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    comparesignal2.SignalSamplesAreEqual(expected_output_file_path, my_output)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def remove_dct():
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    x, y, signal_type = read_signal_from_file(file_path)
    values = []
    sum = 0
    for i in range(len(x)):
        sum = sum + y[i]
    avg = sum / len(x)
    for j in range(len(x)):
        values.append(y[j] - avg)
    expected_output_file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    comparesignal2.SignalSamplesAreEqual(expected_output_file_path, values)


def smoothing_gui():
    smoothing_window = Toplevel(root)
    smoothing_window.title("Enter The Window Size")
    # smoothing_window.geometry("300x200")
    number_of_points_label = Label(smoothing_window, text="Number of points:")
    number_of_points_label.grid(row=0, column=0)
    global number_of_points_entry
    number_of_points_entry = Entry(smoothing_window)
    number_of_points_entry.grid(row=0, column=1)
    compute_smoothing_button = Button(smoothing_window, text="Compute Smoothing", command=smoothing)
    compute_smoothing_button.grid(row=1, column=1)


def smoothing():
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    x, y, signal_type = read_signal_from_file(file_path)
    output_average = []
    number_of_points = int(number_of_points_entry.get())
    for i in range(number_of_points - 1, len(x)):
        temp = y[i]
        for j in range(1, number_of_points):
            temp = temp + y[i - j]
        average = temp / number_of_points
        output_average.append(average)
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    comparesignal2.SignalSamplesAreEqual(file_path, output_average)


def delaying_and_advancing_signal_gui():
    delaying_and_advancing_signal_window = Toplevel(root)
    delaying_and_advancing_signal_window.title("Enter The Steps Shift")
    # smoothing_window.geometry("300x200")
    steps_label = Label(delaying_and_advancing_signal_window, text="Steps:")
    steps_label.grid(row=0, column=0)
    global steps_entry
    steps_entry = Entry(delaying_and_advancing_signal_window)
    steps_entry.grid(row=0, column=1)
    compute_delaying_advancing_button = Button(delaying_and_advancing_signal_window, text="Compute Delay Or Advance",
                                               command=delaying_and_advancing_signal)
    compute_delaying_advancing_button.grid(row=1, column=1)


def delaying_and_advancing_signal(steps=None):
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    x, y, signal_type = read_signal_from_file(file_path)
    output_x = []

    if steps is None:
        steps = int(steps_entry.get())
    for i in x:
        output_x.append(i + steps)
    fig, axs = plt.subplots(2)

    axs[0].plot(x, y)
    axs[0].set_title("Original Signal")
    axs[0].set(xlabel='Sample', ylabel='Amplitude')

    axs[1].plot(output_x, y)
    axs[1].set_title("Shifted Signal")
    axs[1].set(xlabel='Sample', ylabel='Amplitude')

    fig.tight_layout()

    fig.show()
    return output_x


def fold_signal_gui():
    fold_window = Toplevel(root)
    fold_window.title("Enter The Num Folded Times")
    # smoothing_window.geometry("300x200")
    num_folded_times_label = Label(fold_window, text="Number of Folded Times:")
    num_folded_times_label.grid(row=0, column=0)
    global num_folded_times_entry
    num_folded_times_entry = Entry(fold_window)
    num_folded_times_entry.grid(row=0, column=1)
    num_folded_times_button = Button(fold_window, text="Fold Signal",
                                     command=fold_signal)
    num_folded_times_button.grid(row=1, column=1)


def fold_signal(n_folding_times=None):
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    x, y, signal_type = read_signal_from_file(file_path)
    output_y = []
    if n_folding_times is None:
        n_folding_times = int(num_folded_times_entry.get())
    if n_folding_times % 2 != 0:
        for i in range(len(y)):
            output_y.append(y[len(y) - i - 1])
    else:
        output_y = y

    # ~ plot ~
    fig, axs = plt.subplots(2)
    axs[0].plot(x, y)
    axs[0].set_title("Original Signal")
    axs[0].set(xlabel='Sample', ylabel='Amplitude')

    axs[1].plot(x, output_y)
    axs[1].set_title("reversed Signal")
    axs[1].set(xlabel='Sample', ylabel='Amplitude')

    fig.tight_layout()
    fig.show()

    # ~ save in file ~~~
    with open('TestCases/Shifting and Folding/folding_function_output.txt', 'w') as file:
        file.write("0\n")
        file.write("0\n")
        file.write(f"{len(x)}\n")

        for i in range(len(output_y)):
            file.writelines(f"{x[i]} {output_y[i]}\n")
    return output_y


def shift_folded_signal_gui():
    shift_folded_window = Toplevel(root)
    shift_folded_window.title("Shifted Folded Window")
    steps_label = Label(shift_folded_window, text="Steps:")
    steps_label.grid(row=0, column=0)
    global steps_entry
    steps_entry = Entry(shift_folded_window)
    steps_entry.grid(row=0, column=1)
    num_folded_times_label = Label(shift_folded_window, text="Number of Folded Times:")
    num_folded_times_label.grid(row=1, column=0)
    global num_folded_times_entry
    num_folded_times_entry = Entry(shift_folded_window)
    num_folded_times_entry.grid(row=1, column=1)
    shift_folded_button = Button(shift_folded_window, text="Fold Signal",
                                 command=shift_folded_signal)
    shift_folded_button.grid(row=2, column=1)


def shift_folded_signal(n_folding_times=None, steps=None):
    if n_folding_times is None:
        n_folding_times = int(num_folded_times_entry.get())
    if steps is None:
        steps = int(steps_entry.get())
    samples = fold_signal(n_folding_times)
    indecies = delaying_and_advancing_signal(steps)
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    Shift_Fold_Signal.Shift_Fold_Signal(file_path, indecies, samples)


def dft2(x_Of_k=None):
    x_of_k = []
    global amplituedes
    global phase_shifts
    amplituedes = []
    phase_shifts = []
    if x_Of_k is None:
        file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        x, y, signal_type = read_signal_from_file(file_path)
        for k in range(len(x)):
            count = 0
            for n in range(len(x)):
                # 1st harmonic x(0) ######################################
                if n * k == 0:
                    count = count + y[n]
                    count = complex(count)
                # other n harmonics ######################################
                else:
                    degree = (2 * math.pi * n * k) / len(x)
                    cos = round(math.cos(degree), 15)
                    sin = round(math.sin(degree), 15)
                    exp = complex(cos, -sin)
                    count = count + y[n] * exp
            # append x(k) in the list of harmonics ########################
            x_of_k.append(count)
    else:
        x_of_k = x_Of_k
    # compute frequencies ,amplitudes and phase shifts #########################################################
    sampling_freq = float(sampling_frequency_entry.get())
    fundamental_freq = (2 * np.pi * sampling_freq) / len(x_of_k)
    frequencies = []
    for a in range(len(x_of_k)):
        # Amplitude ######################################
        amp = math.sqrt(math.pow(x_of_k[a].real, 2) + math.pow(x_of_k[a].imag, 2))
        amplituedes.append(amp)
        # Phase Shift ####################################
        phase = np.arctan2(x_of_k[a].imag, x_of_k[a].real)
        phase_shifts.append(phase)
        # Frequency ######################################
        frequencies.append(fundamental_freq * (a + 1))

    # ~~~ save Amplitude and Phase in text file ~~~~~~
    with open('my_output.txt', 'w') as f:
        f.write("0")
        f.write("\n")
        f.write("1")
        f.write("\n")
        f.write(str(len(phase_shifts)))
        for m, n in zip(amplituedes, phase_shifts):
            m = str(m)
            n = str(n)
            f.write("\n")
            f.write(m)
            f.write(" ")
            f.write(n)
    return amplituedes, phase_shifts


def idft2(amp, phase):
    x_of_k = []
    global x_of_n
    x_of_n = []
    complex_numbers = []
    # convert the Amplitude and Phase Shift to the complex number form "real and imaginary"#####################
    for i in range(len(amp)):
        # real = Amplitude * cos(phaseShift) ##########################
        real_number = amp[i] * math.cos(phase[i])
        # imaginary = Amplitude * sin(phaseShift) #####################
        imaginary_number = amp[i] * math.sin(phase[i])
        complex_number = complex(real_number, imaginary_number)
        complex_numbers.append(complex_number)
    # x(k) are complex numbers ########################################
    for n in range(len(amp)):
        count = 0
        for k in range(len(amp)):
            # 1st harmonic x(0) ######################################
            if n * k == 0:
                count = count + complex_numbers[k]

            # other n harmonics ######################################
            else:
                degree = (2 * math.pi * n * k) / len(amp)
                cos = round(math.cos(degree), 12)
                sin = round(math.sin(degree), 12)
                exp = complex(cos, sin)
                count = count + (complex_numbers[k] * exp)

        count_real = count.real / len(amp)
        count_imagine = round(count.imag, 11)
        x_of_n.append(count_real)
        x_of_k.append(complex(count_real, count_imagine))

    return x_of_n

def dft3(y):
    x_of_k = []
    for k in range(len(y)):
        count = 0
        for n in range(len(y)):
            # 1st harmonic x(0) ######################################
            if n * k == 0:
                count = count + y[n]
                count = complex(count)
            # other n harmonics ######################################
            else:
                degree = (2 * math.pi * n * k) / len(y)
                cos = round(math.cos(degree), 15)
                sin = round(math.sin(degree), 15)
                exp = complex(cos, -sin)
                count = count + y[n] * exp
        # append x(k) in the list of harmonics ########################
        x_of_k.append(count)

    return x_of_k

def remove_dc_component_task_6(temp=None):
    if temp is None:
        temp_fn()
    else:
        amp, phase = dft2()
        amp[0] = 0
        phase[0] = 0
        x_of_n = idft2(amp, phase)
        for i in range(len(x_of_n)):
            x_of_n[i] = round(x_of_n[i], 3)
        file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        comparesignal2.SignalSamplesAreEqual(file_path, x_of_n)
        print(x_of_n)


def temp_fn():
    sampling_frequency_window = Toplevel(root)
    sampling_frequency_window.title("Enter Sampling Frequency in HZ")

    sampling_frequency_label = Label(sampling_frequency_window, text="Sampling Frequency:")
    sampling_frequency_label.grid(row=0, column=0)

    global sampling_frequency_entry
    sampling_frequency_entry = Entry(sampling_frequency_window)
    sampling_frequency_entry.grid(row=0, column=1)

    enter_button = Button(sampling_frequency_window, text="Enter")
    enter_button.grid(row=1, column=1)

    enter_button = Button(sampling_frequency_window, text="Enter", command=temp_fn2)
    enter_button.grid(row=1, column=1)


def temp_fn2():
    remove_dc_component_task_6("pass")


def convolution():
    indecies = []
    # ~~~~~~ Read Signal 1 ~~~~~~~
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    x1, y1, signal_type = read_signal_from_file(file_path)
    # ~~~~~~ Read Signal 2 ~~~~~~~
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    x2, y2, signal_type = read_signal_from_file(file_path)

    begin = x1[0] + x2[0]
    last = x1[len(x1) - 1] + x2[len(x2) - 1]

    begin = int(begin)
    last = int(last)

    for i in range(begin, last + 1):
        indecies.append(i)
    output_y = np.zeros(len(indecies), dtype=int)
    for j in range(len(y1)):
        for k in range(len(y2)):
            output_y[j + k] += int(y1[j] * y2[k])
    print(output_y)
    ConvTest.ConvTest(indecies, output_y)


def compute_normalized_correlation():
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    _, y1, _ = read_signal_from_file(file_path)
    file_path2 = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    _, y2, _ = read_signal_from_file(file_path2)

    def compute_normalized_correlation(list1, list2):
        size = len(list1)
        normalized_corr = []
        sum_squared = (sum(x ** 2 for x in list1)) * (sum(x ** 2 for x in list2))
        for i in range(len(list1)):
            total = 0
            for j in range(len(list2)):
                # index is a circular shift of the second list
                index = (j + i) % size
                total += list1[j] * list2[index]

            average = total / size
            normalized_corr.append(average)
        output = [round(x / (math.sqrt(sum_squared) / size), 8) for x in normalized_corr]

        print("Result:", output)
        filename_output = filedialog.askopenfilename(title="Select a Signal File")
        comparesignal2.SignalSamplesAreEqual(filename_output, output)

    compute_normalized_correlation(y1, y2)


def fast_convolution():
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    x1, y1, _ = read_signal_from_file(file_path)
    file_path2 = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    x2, y2, _ = read_signal_from_file(file_path2)
    n1, n2 = len(y1), len(y2)
    # append with zeroes
    new_size = n1 + n2 - 1
    signal1, signal2 = [], []

    indecies = []
    begin = x1[0] + x2[0]
    last = x1[len(x1) - 1] + x2[len(x2) - 1]

    begin = int(begin)
    last = int(last)

    for i in range(begin, last + 1):
        indecies.append(i)
    for i in range(new_size):
        if i in range(n1):
            signal1.append(y1[i])
        else:
            signal1.append(0)

        if i in range(n2):
            signal2.append(y2[i])
        else:
            signal2.append(0)
    # convert to frequency domain
    signal1_harmonics = dft3(signal1)
    signal2_harmonics = dft3(signal2)
    mult = [signal1_harmonics[i] * signal2_harmonics[i] for i in range(len(signal2_harmonics))]
    result = idft3(mult)
    result = [round(result[i]) for i in range(len(result))]
    ConvTest.ConvTest(indecies, result)

def Fast_Correlation():
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    x1, y1, _ = read_signal_from_file(file_path)
    file_path2 = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    x2, y2, _ = read_signal_from_file(file_path2)
    len(y1)
    frequecy_domain = []
    result = []
    y_conjugates = []
    complex_y1 = dft3(y1)
    complex_y2 = dft3(y2)
    for i in complex_y1:
        y_cong_value = i.conjugate()
        y_conjugates.append(y_cong_value)
    for i in range(len(complex_y1)):
        frequecy_domain.append(y_conjugates[i] * complex_y2[i])
    time_domain = idft3(frequecy_domain)

    for i in range(len(y1)):
        result.append(time_domain[i] / len(y1))
    indeceis = x1
    file_path3 = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    CompareSignal.Compare_Signals(file_path3, indeceis, result)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ General GUI ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def task1_gui():
    task1_window = Toplevel(root)
    task1_window.title("Task 1")
    task1_window.geometry("300x140")

    generate_signal_from_file_button = Button(task1_window, text="Generate Signal File", bg="black", fg="white",
                                              height=3, width=20,
                                              command=generate_wave_from_file)
    generate_signal_from_file_button.pack(pady=5)

    generate_signal_wave_button = Button(task1_window, text="Generate Signal Wave", bg="black", fg="white", height=3,
                                         width=20,
                                         command=signal_generation_menu_gui)
    generate_signal_wave_button.pack(pady=5)


def task2_gui():
    task2_window = Toplevel(root)
    task2_window.title("Task 2")
    task2_window.geometry("500x400")

    add_two_signal_waves_button = Button(task2_window, text="Add Two Signal Waves", bg="black", fg="white", height=3,
                                         width=25,
                                         command=add_signals_gui)
    add_two_signal_waves_button.pack(pady=5)

    subtract_two_signal_waves_button = Button(task2_window, text="Subtract Two Signal Waves", bg="black", fg="white",
                                              height=3, width=25,
                                              command=sub_signals_gui)
    subtract_two_signal_waves_button.pack(pady=5)

    multiply_signal_constant_button = Button(task2_window, text="Multiply Signal By A Constant", bg="black", fg="white",
                                             height=3, width=25, command=multiply_by_constant_gui)
    multiply_signal_constant_button.pack(pady=5)

    square_signal_button = Button(task2_window, text="Square Signal Wave", bg="black", fg="white", height=3, width=25,
                                  command=square_signal_gui)
    square_signal_button.pack(pady=5)

    shift_signal_button = Button(root, text="Shift Signal Wave", bg="black", fg="white", height=3, width=25,
                                 command=shift_signal_gui)
    shift_signal_button.pack(pady=5)

    normalize_signal_button = Button(task2_window, text="Normalize Signal Wave", bg="black", fg="white", height=3,
                                     width=25,
                                     command=normalize_signal_gui)
    normalize_signal_button.pack(pady=5)

    accumulative_sum_signal_button = Button(task2_window, text="Accumulative Sum Signal Wave", bg="black", fg="white",
                                            height=3, width=25
                                            , command=accumulative_sum_gui)
    accumulative_sum_signal_button.pack(pady=5)


def task3_gui():
    task3_window = Toplevel(root)
    task3_window.title("Task 3")
    task3_window.geometry("300x100")

    quantize_signal_button = Button(task3_window, text="Quantize Signal Wave", bg="black", fg="white", height=3,
                                    width=25,
                                    command=quantize_signal_gui)
    quantize_signal_button.pack(pady=5)


def task4_gui():
    task4_window = Toplevel(root)
    task4_window.title("Task 4")
    task4_window.geometry("300x100")

    fourier_transform_button = Button(task4_window, text="Fourier transform", bg="black", fg="white", height=3,
                                      width=25,
                                      command=fourier_transform_gui)
    fourier_transform_button.pack(pady=3)


def task5_gui():
    task5_window = Toplevel(root)
    task5_window.title("Task 5")
    task5_window.geometry("300x100")

    DCT_button = Button(task5_window, text="Compute DCT", bg="black", fg="white", height=3, width=25, command=DCT_gui)
    DCT_button.pack(pady=3)


def task6_gui():
    task6_window = Toplevel(root)
    task6_window.title("Task 6")
    task6_window.geometry("500x400")

    Smoothing_button = Button(task6_window, text="Smoothing", bg="black", fg="white", height=3, width=25,
                              command=smoothing_gui)
    Smoothing_button.pack(pady=5)

    remove_dct_button = Button(task6_window, text="Remove DC Component", bg="black", fg="white", height=3, width=25,
                               command=remove_dc_component_task_6)
    remove_dct_button.pack(pady=5)

    delaying_and_advancing_button = Button(task6_window, text="Delaying and Advancing", bg="black", fg="white",
                                           height=3, width=25
                                           , command=delaying_and_advancing_signal_gui)
    delaying_and_advancing_button.pack(pady=5)

    fold_button = Button(task6_window, text="Fold the Signal", bg="black", fg="white", height=3, width=25,
                         command=fold_signal_gui)
    fold_button.pack(pady=5)

    Shift_Fold_Signal_button = Button(task6_window, text="Shift and Fold signal", bg="black", fg="white", height=3,
                                      width=25,
                                      command=shift_folded_signal_gui)
    Shift_Fold_Signal_button.pack(pady=5)

    derivative_button = Button(task6_window, text="Compute Derivative", bg="black", fg="white", height=3, width=25,
                               command=DerivativeSignal.DerivativeSignal)
    derivative_button.pack(pady=5)


def task7_gui():
    task7_window = Toplevel(root)
    task7_window.title("Task 7")
    task7_window.geometry("300x100")

    convolution_button = Button(task7_window, text="Convolution", bg="black", fg="white", height=3,
                                width=25,
                                command=convolution)
    convolution_button.pack(pady=3)

def task9_gui():
    task9_window = Toplevel(root)
    task9_window.title("Task 7")
    task9_window.geometry("300x100")

    convolution_button = Button(task9_window, text="Convolution", bg="black", fg="white", height=3,width=25,
                                command=fast_convolution)
    convolution_button.pack(pady=3)
    correlation_button = Button(task9_window, text="Correlation", bg="black", fg="white", height=3, width=25,
                                command=Fast_Correlation)
    correlation_button.pack(pady=3)


# Create window for the signal generation (GUI)
root = Tk()
root.title("Signal Processing Framework")
root.geometry("500x600")

Task1_button = Button(root, text="Task 1", bg="black", fg="white", height=3, width=20, command=task1_gui)
Task1_button.pack(pady=5)

Task2_button = Button(root, text="Task 2", bg="black", fg="white", height=3, width=20, command=task2_gui)
Task2_button.pack(pady=5)

Task3_button = Button(root, text="Task 3", bg="black", fg="white", height=3, width=20, command=task3_gui)
Task3_button.pack(pady=5)

Task4_button = Button(root, text="Task 4", bg="black", fg="white", height=3, width=20, command=task4_gui)
Task4_button.pack(pady=5)

Task5_button = Button(root, text="Task 5", bg="black", fg="white", height=3, width=20, command=task5_gui)
Task5_button.pack(pady=5)

Task6_button = Button(root, text="Task 6", bg="black", fg="white", height=3, width=20, command=task6_gui)
Task6_button.pack(pady=5)

Task7_button = Button(root, text="Task 7", bg="black", fg="white", height=3, width=20, command=task7_gui)
Task7_button.pack(pady=5)

Task8_button = Button(root, text="Task 8", bg="black", fg="white", height=3, width=20, command=compute_normalized_correlation)
Task8_button.pack(pady=5)

Task9_button = Button(root, text="Task 9", bg="black", fg="white", height=3, width=20, command=task9_gui)
Task9_button.pack(pady=5)


# Run the main event loop
root.mainloop()