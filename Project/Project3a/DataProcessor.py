import numpy as np

def apply_window(length, slide, arm, wrist, detection):
    index = 0
    new_arm = []
    new_wrist = []
    new_detect = []
    
    while True:
        arm_row = arm[index]
        wrist_row = wrist[index]
        for i in range(1,length):
            arm_row = np.concatenate((arm_row, arm[index+i]))
            wrist_row = np.concatenate((wrist_row, wrist[index+i]))
        new_arm.append(arm_row)
        new_wrist.append(wrist_row)
        new_detect.append(round(np.mean(detection[index:(index+length)])))
        index += slide
        if index+length-1 >= len(arm):
          break
    new_arm = np.array(new_arm)
    new_wrist = np.array(new_wrist)
    new_detect = np.array(new_detect)
    return new_arm, new_wrist, new_detect

def expand_predictions(old_detection, slide, pred):
    new_pred = []
    for i in range(len(pred)):
        val = pred[i]
        for j in range(slide):
            new_pred.append(val)
    remainder = len(old_detection) - len(new_pred)
    for i in range(remainder):
        new_pred.append(pred[-1])
    return np.array(new_pred)

def expand_predictions_2(old_detection, length, slide, pred):
    new_pred = []

    #Initialize with first (length-slide) values to predictions are added to the end of the window
    val = pred[0]
    for i in range(length-slide):
        new_pred.append(val)
    
    for i in range(len(pred)):
        val = pred[i]
        for j in range(slide):
            new_pred.append(val)
    remainder = len(old_detection) - len(new_pred)
    for i in range(remainder):
        new_pred.append(pred[-1])
    return np.array(new_pred)