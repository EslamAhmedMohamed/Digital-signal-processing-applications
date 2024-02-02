import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import filedialog
from tkinter.ttk import Treeview
import QuanTest1
import QuanTest2
import math
import DFTtest
import IDFTtest
import signalcompare

def read_signal_from_file(file_path):
    # Read signal samples from a text file
    with open(file_path) as file:
        listOfLines = file.readlines()
        IsPeriodic=listOfLines[0]
        SignalType=listOfLines[1]
        SignalType=int(SignalType.strip())
        sampleSize=listOfLines[2]
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
    x, y = read_signal_from_file(file_path)
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
            signal2 = generate_cosinusoidal_signal(amplitude, phase_shift, analog_frequency, sampling_frequency, duration)
            title = f"Cosine Wave: A={amplitude}, θ={phase_shift}, f={analog_frequency}, fs={sampling_frequency}"
            plot_signal_wave(signal2, title, flag1, flag2)

    generate_signal_button = Button(generation_window, text="Generate Signal Wave", bg="black", fg="white", command=generate_wave)
    generate_signal_button.grid(row=6, column=1)


def add_signals_gui():
    sig_num_window = Toplevel(root)
    sig_num_window.title("Enter Number Of Signals")

    signal_num_label = Label(sig_num_window, text="Number of signals:")
    signal_num_label.grid(row=0, column=0)
    signal_num_entry = Entry(sig_num_window)
    signal_num_entry.grid(row=0, column=1)

    def add_waves():
        num_of_signals = int(signal_num_entry.get())
        input_y = []
        output_y = []
        for _ in range(num_of_signals):
            file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
            x, y = read_signal_from_file(file_path)
            input_y.append(y)

        output_y = [sum(signal_values) for signal_values in zip(*input_y)]

        plt.figure()
        plt.plot(x, output_y)
        plt.title("Output Signal")
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.show()

    enter_button = Button(sig_num_window, text="Enter", command=add_waves)
    enter_button.grid(row=1, column=1)


def sub_signals_gui():
    output_y = []

    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    x1, y1 = read_signal_from_file(file_path)
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    x2, y2 = read_signal_from_file(file_path)
    output_y = [amp1 - amp2 for amp1, amp2 in zip(y1, y2)]

    fig, axs = plt.subplots(3, sharex=True, sharey=True)
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
    output_y = []
    constant_window = Toplevel(root)
    constant_window.title("Enter Your Constant")

    constant_label = Label(constant_window, text="Constant:")
    constant_label.grid(row=0, column=0)
    constant_entry = Entry(constant_window)
    constant_entry.grid(row=0, column=1)

    def multiply_signal():
        file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        x, y = read_signal_from_file(file_path)

        const = float(constant_entry.get())
        for i in y:
            output_y.append(i*const)
        fig, axs = plt.subplots(2)

        axs[0].plot(x, y)
        axs[0].set_title("Original Signal")
        axs[0].set(xlabel='Sample', ylabel='Amplitude')

        axs[1].plot(x, output_y)
        axs[1].set_title("Edited Signal")
        axs[1].set(xlabel='Sample', ylabel='Amplitude')

        fig.tight_layout()

        fig.show()
    enter_button = Button(constant_window, text="Enter", command=multiply_signal)
    enter_button.grid(row=1, column=1)


def square_signal_gui():
    output_y = []
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    x, y = read_signal_from_file(file_path)
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
    output_x = []
    constant_window = Toplevel(root)
    constant_window.title("Enter Your Constant")

    constant_label = Label(constant_window, text="Constant:")
    constant_label.grid(row=0, column=0)
    constant_entry = Entry(constant_window)
    constant_entry.grid(row=0, column=1)

    def shift_signal():
        file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        x, y = read_signal_from_file(file_path)

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

    enter_button = Button(constant_window, text="Enter", command=shift_signal)
    enter_button.grid(row=1, column=1)


def normalize_signal_gui():
    output_y = []
    normalize_type_window = Toplevel(root)
    normalize_type_window.title("Enter Valid Normalize Type")

    choice_label = Label(normalize_type_window, text="Type (-1 to 1 || 0 to 1):")
    choice_label.grid(row=0, column=0)
    choice_entry = Entry(normalize_type_window)
    choice_entry.grid(row=0, column=1)

    def normalize_signal():
        file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        x, y = read_signal_from_file(file_path)
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

    enter_button = Button(normalize_type_window, text="Enter", command=normalize_signal)
    enter_button.grid(row=1, column=1)


def accumulative_sum_gui():
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    x, y = read_signal_from_file(file_path)
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
    r1 = Radiobutton(quantize_window, text="Number Of Levels", variable=radio_button, value=1, command=num_of_levels_fun)
    r1.pack(anchor=W)

    r2 = Radiobutton(quantize_window, text="Number Of Bits", variable=radio_button, value=2, command=num_of_bits_fun)
    r2.pack(anchor=W)

    def quantize_signal():
######### declarations ######

        file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        x, y = read_signal_from_file(file_path)
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

######### calc delta, intervals and midpoints #######

        delta = (max_y - min_y) / num_of_levels
        new_max = min_y + delta
        while min_y < max_y:
            intervals.append([min_y, new_max])
            min_y = round((min_y + delta),3)
            new_max = round((min_y + delta),3)

        for i in intervals:
            midpoints.append(round((i[0] + i[1]) / 2,3))

        quantized_values = []
        def find_interval(inpt):
            count = 0
            for k in range(len(intervals)):
                min_val = intervals[k][0]
                max_val = intervals[k][1]
                if min_val <= inpt:
                    if inpt <= max_val:
                        num_of_levels_of_var = count
                        indices.append(num_of_levels_of_var+1)
                        quantized_values.append(midpoints[num_of_levels_of_var])
                        encoded_values.append(format(num_of_levels_of_var, f'0{num_of_bits}b'))
                        break
                count = count + 1
            return num_of_levels_of_var


######### fitch the interval of the input ######

        for inpt in y:
            interval_number = find_interval(inpt)
            quantized_error.append(round(midpoints[interval_number] - inpt,3))
        squared_error = []
        for m in quantized_error:
            squared_error.append(round(m*m,3))
        #squared_error = squared_error / len(quantized_error)

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

        if radio_button.get() == 2:
            QuanTest1.QuantizationTest1('Quan1_Out.txt', encoded_values, quantized_values)
        elif radio_button.get() == 1:
            QuanTest2.QuantizationTest2('Quan2_Out.txt', indices, encoded_values, quantized_values, quantized_error)


def fourier_transform_gui():
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    x, y, signal_type = read_signal_from_file(file_path)

    def fourier_transform():

        def dft():
            x_of_k = []
            global amplituedes
            global phase_shifts
            amplituedes = []
            phase_shifts = []
            for k in range(len(x)):
                count = 0
                for n in range(len(x)):
                    if n * k == 0:
                        count = count + y[n]
                        count = complex(count)
                    else:
                        degree1 = (2 * 180 * n * k) / len(x)
                        degree = (2 * math.pi * n * k) / len(x)
                        cos = round(math.cos(degree), 15)
                        sin = round(math.sin(degree), 15)
                        exp = complex(cos, -sin)
                        count = count + y[n] * exp
                x_of_k.append(count)
            sampling_freq = float(sampling_frequency_entry.get())
            fundamental_freq = (2 * np.pi * sampling_freq) / len(x)
            frequencies = []
            for a in range(len(x_of_k)):
                amp = math.sqrt(math.pow(x_of_k[a].real, 2) + math.pow(x_of_k[a].imag, 2))
                amplituedes.append(amp)
                phase = np.arctan2(x_of_k[a].imag, x_of_k[a].real)
                phase_shifts.append(phase)
                frequencies.append(fundamental_freq * (a + 1))

            with open('my_output.txt', 'w') as f:
                f.write("0")
                f.write("\n")
                f.write(str(signal_type))
                f.write("\n")
                f.write(str(len(phase_shifts)))
                for m, n in zip(amplituedes, phase_shifts):
                    m = str(m)
                    n = str(n)
                    f.write("\n")
                    f.write(m)
                    f.write(" ")
                    f.write(n)

            fig, axs = plt.subplots(2)
            axs[0].stem(frequencies, amplituedes)
            axs[0].set_title("Frequencies vs Amplitudes")
            axs[0].set(xlabel='Frequency', ylabel='Amplitude')

            axs[1].stem(frequencies, phase_shifts)
            axs[1].set_title("Frequencies vs Phase Shifts")
            axs[1].set(xlabel='Frequency', ylabel='Phase Shift')

            fig.tight_layout()

            fig.show()

        def idft():
            x_of_k = []
            global x_of_n
            x_of_n = []
            complex_numbers = []
            for i in range(len(x)):
                real_number = x[i] * math.cos(y[i])
                imaginary_number = x[i] * math.sin(y[i])
                complex_number = complex(real_number, imaginary_number)
                complex_numbers.append(complex_number)
            for n in range(len(x)):
                count = 0
                for k in range(len(x)):
                    if n * k == 0:
                        count = count + complex_numbers[k]
                    else:
                        degree1 = (2 * 180 * n * k) / len(x)
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

        if signal_type == 0:
            dft()

            file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
            correct_amp, correct_phase = read_signal_from_file(file_path)
            bool1 = signalcompare.SignalComapreAmplitude(amplituedes, correct_amp)
            bool2 = signalcompare.SignalComaprePhaseShift(phase_shifts, correct_phase)
            print(bool1, bool2)

        else:
            idft()
            file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
            correct_y = IDFTtest.output_fun(file_path)
            IDFTtest.my_output_fun(x_of_n, correct_y)

    if signal_type == 0:
        sampling_frequency_window = Toplevel(root)
        sampling_frequency_window.title("Enter Sampling Frequency in HZ")

        sampling_frequency_label = Label(sampling_frequency_window, text="Sampling Frequency:")
        sampling_frequency_label.grid(row=0, column=0)
        global sampling_frequency_entry
        sampling_frequency_entry = Entry(sampling_frequency_window)
        sampling_frequency_entry.grid(row=0, column=1)
        enter_button = Button(sampling_frequency_window, text="Enter", command=fourier_transform)
        enter_button.grid(row=1, column=1)
    else:
        fourier_transform()



# Create window for the signal generation (GUI)
root = Tk()
root.title("Signal Processing Framework")
root.geometry("700x700")


generate_signal_from_file_button = Button(root, text="Generate Signal File", bg="black", fg="white", padx=10, pady=10, command=generate_wave_from_file)
generate_signal_from_file_button.pack(pady=3)

generate_signal_wave_button = Button(root, text="Generate Signal Wave", bg="black", fg="white", padx=10, pady=10, command=signal_generation_menu_gui)
generate_signal_wave_button.pack(pady=3)

add_two_signal_waves_button = Button(root, text="Add Two Signal Waves", bg="black", fg="white", padx=10, pady=10, command=add_signals_gui)
add_two_signal_waves_button.pack(pady=3)

subtract_two_signal_waves_button = Button(root, text="Subtract Two Signal Waves", bg="black", fg="white", padx=10, pady=10, command=sub_signals_gui)
subtract_two_signal_waves_button.pack(pady=3)

multiply_signal_constant_button = Button(root, text="Multiply Signal By A Constant", bg="black", fg="white", padx=10, pady=10, command=multiply_by_constant_gui)
multiply_signal_constant_button.pack(pady=3)

square_signal_button = Button(root, text="Square Signal Wave", bg="black", fg="white", padx=10, pady=10, command=square_signal_gui)
square_signal_button.pack(pady=3)

shift_signal_button = Button(root, text="Shift Signal Wave", bg="black", fg="white", padx=10, pady=10, command=shift_signal_gui)
shift_signal_button.pack(pady=3)

normalize_signal_button = Button(root, text="Normalize Signal Wave", bg="black", fg="white", padx=10, pady=10, command=normalize_signal_gui)
normalize_signal_button.pack(pady=3)

accumulative_sum_signal_button = Button(root, text="Accumulative Sum Signal Wave", bg="black", fg="white", padx=10, pady=10, command=accumulative_sum_gui)
accumulative_sum_signal_button.pack(pady=3, padx=150)

quantize_signal_button = Button(root, text="Quantize Signal Wave", bg="black", fg="white", padx=10, pady=10, command=quantize_signal_gui)
quantize_signal_button.pack(pady=3)

quantize_signal_button = Button(root, text="Fourier transform", bg="black", fg="white", padx=10, pady=10, command=fourier_transform_gui)
quantize_signal_button.pack(pady=3)

# Run the main event loop
root.mainloop()
