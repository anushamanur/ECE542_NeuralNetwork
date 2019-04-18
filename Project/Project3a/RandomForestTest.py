import DataReader as dr
import DataProcessor as dp
import DataAnalyzer as da
import DataManager as dm
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.neural_network import MLPClassifier 
  

def get_accuracy_windowed(classifier, parameters, length, slide, arms, wrists, detections):
    #Assemble training data
    armIMU = arms[0]
    wristIMU = wrists[0]
    detection = detections[0]
    for i in range(1,5):
        armIMU = np.concatenate((armIMU, arms[i]), axis=0)
        wristIMU = np.concatenate((wristIMU, wrists[i]), axis=0)
        detection = np.concatenate((detection, detections[i]), axis=0)
        
    train_arm, train_wrist, train_detect = dp.apply_window(length, slide, armIMU, wristIMU, detection)
    print("Window applied to training data")
    
    #Assemble validation data
    arm, wrist, time, detect = dr.get_data(6)
    detect = detect.astype(int)
    detect = detect.ravel()
    val_arm, val_wrist, val_detect = dp.apply_window(length, slide, arm, wrist, detect)
    print("Window applied to validation data")

    train_arm_wrist = np.concatenate((train_arm, train_wrist), axis=1)
    val_arm_wrist = np.concatenate((val_arm, val_wrist), axis=1)
    
    accuracies = perform_classification(classifier, 
                                        parameters, 
                                        slide, 
                                        train_arm_wrist, 
                                        train_detect, 
                                        val_arm_wrist, 
                                        val_detect)
    
    return accuracies

def get_accuracy_features(classifier, parameters, length, slide):
    data = dm.load_windowed_data(length, slide)
    train_arm = data["train_arm"]
    train_wrist = data["train_wrist"]
    train_detect = data["train_detect"]
    val_arm = data["test_arm"]
    val_wrist = data["test_wrist"]
    #val_detect = data["test_detect"]

    train_arm_wrist = np.concatenate((train_arm, train_wrist), axis=1)
    val_arm_wrist = np.concatenate((val_arm, val_wrist), axis=1)
    
    _, _, _, val_detect = dr.get_data(6)
    val_detect = val_detect.astype(int)
    val_detect = val_detect.ravel()
    
    accuracies = perform_classification(classifier, 
                                        parameters, 
                                        slide, 
                                        train_arm_wrist, 
                                        train_detect, 
                                        val_arm_wrist, 
                                        val_detect,length)
    
    return accuracies
    
def perform_classification(clf, parameters, slide, train_arm_wrist, train_detect, val_arm_wrist, val_detect,length):
    accuracies = []
    parameters = parameters[clf]

    for parameter in parameters:
        #Train random forest classifier
        classifier = RandomForestClassifier(n_estimators = parameter)
        
        if clf is "knn":
            #Train k-nearest neighbors classifier
            classifier = KNeighborsClassifier(n_neighbors=parameter) 
        
        elif clf is "mlp":
            #Train MLP classifier
            classifier = MLPClassifier(hidden_layer_sizes= ((100),)*parameter, max_iter=1000)
	
            
        
        #Fit classifier to training data
        classifier.fit(train_arm_wrist, train_detect)
        
        predictions = classifier.predict(val_arm_wrist)
        predictions = dp.expand_predictions_2(val_detect, length, slide, predictions)
        
        errors = abs(predictions - val_detect)
        print('Mean Absolute Error: ' + str(np.mean(errors)))
        accuracies.append(1-np.mean(errors))
    return accuracies

#arms, wrists, detections = dr.get_all_data()
lengths = [100, 150, 200, 250, 300, 350, 400]
#slides = [0.1, 0.25]
rf_estimators = [100, 150, 200, 250, 300, 350, 400]
knn_neighbors = [50, 100, 150, 200, 300, 400, 500]
mlp_hiddenlayers=[2,3,5,10, 20]


classifiers = ['rf', 'knn','mlp']

parameters = {'rf':rf_estimators,
              'knn':knn_neighbors,
	      'mlp':mlp_hiddenlayers
             }

accuracies = []
for length in lengths:
    slide = 50
    acc_row = get_accuracy_features("mlp", parameters, length, slide)
    #acc_row = get_accuracy_windowed("rf", parameters, length, slide, arms, wrists, detections)
    accuracies.append(acc_row)
print(accuracies)
acc = np.array(accuracies)
