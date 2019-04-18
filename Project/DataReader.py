import numpy as np

DATA_PATH = '/content/gdrive/My Drive/ECE542Project3A-master/AllData/'

files = ["armIMU", "wristIMU", "time", "detection"]

def get_data(number):
    path = DATA_PATH
    armIMU = []
    wristIMU = []
    time = []
    detection = []

    data = [armIMU, wristIMU, time, detection]
    for i in range(len(files)):
        file_name = path + files[i] + str(number) + ".txt"
        print("Reading: " + str(file_name))
        data[i] = read_lines(file_name)
    return tuple(data)
    
def read_lines(file):
    #Returns text file lines as a numpy array
    text_data = []
    float_data = []

    #Read in data as text
    with open(file) as textFile:
        for line in textFile:
            text_data.append(line)
            
    #Convert data to list of floats
    for line in text_data:
        line = line.split()
        line = [float(number) for number in line]
        float_data.append(line)
    return np.array(float_data)
    
def get_all_data():
    arms = []
    wrists = []
    detections = []
    for i in range(1,6):
        armIMU, wristIMU, time, detection = get_data(i)
        detection = detection.astype(int)
        detection = detection.ravel()
        
        arms.append(armIMU)
        wrists.append(wristIMU)
        detections.append(detection)
    return arms, wrists, detections
