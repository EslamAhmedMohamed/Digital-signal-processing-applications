
def output_fun(file_path):
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
    return samplesX, samplesY


def my_output_fun(my_amplitude, my_phase_shift, correct_amplitude, correct_phase_shift):
    for i in range(len(correct_amplitude)):
        if abs(my_amplitude[i] - correct_amplitude[i]) < 0.01:
            continue
        else:
            print("DFT Test case failed, your Amplitude have different values from the expected one")

        if abs(my_phase_shift[i] - correct_phase_shift[i]) < 0.01:
            continue
        else:
            print("DFT Test case failed, your Phase Shift have different values from the expected one")
            return
    print("DFT Test case passed successfully")
