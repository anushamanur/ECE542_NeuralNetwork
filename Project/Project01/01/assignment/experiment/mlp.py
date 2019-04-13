#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: mlp.py
# Author: Qian Ge <qge2@ncsu.edu>

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../')
import src.network2 as network2
import src.mnist_loader as loader
import src.activation as act

DATA_PATH = '../../../data/' #D:\\
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', action='store_true',
                        help='Check data loading.')
    parser.add_argument('--sigmoid', action='store_true',
                        help='Check implementation of sigmoid.')
    parser.add_argument('--gradient', action='store_true',
                        help='Gradient check')
    parser.add_argument('--train', action='store_true',
                        help='Train the model')
    parser.add_argument('--test', action='store_true',
                        help='Load the model and test')


    return parser.parse_args()

def load_data():
    train_data, valid_data, test_data = loader.load_data_wrapper(DATA_PATH)
    print('Number of training: {}'.format(len(train_data[0])))
    print('Number of validation: {}'.format(len(valid_data[0])))
    print('Number of testing: {}'.format(len(test_data[0])))
    return train_data, valid_data, test_data

def test_sigmoid():
    z = np.arange(-10, 10, 0.1)
    y = act.sigmoid(z)
    y_p = act.sigmoid_prime(z)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(z, y)
    plt.title('sigmoid')
    plt.figure()
    plt.subplot(1, 2, 2)
    plt.plot(z, y_p)
    plt.title('derivative sigmoid')
    plt.show()


def plot_learning_curve(cost_parameters,num_epochs,len_train,len_val):
    evaluation_cost, evaluation_accuracy, \
            training_cost, training_accuracy=cost_parameters
    epochs_list=[i for i in range(num_epochs)]
    evaluation_accuracy=[(e/len_val)*100 for e in evaluation_accuracy]
    training_accuracy=[(t/len_train)*100 for t in training_accuracy]
    print("Training Accuracy is:",training_accuracy[-1]/100.0)
    print("Validation Accuracy is:",evaluation_accuracy[-1]/100.0)
   
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(epochs_list,evaluation_cost,'b-',epochs_list,training_cost,'r-')
    plt.legend(('Validtion Error','Training Error'))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(epochs_list,evaluation_accuracy,'b-',epochs_list,training_accuracy,'r-')
    plt.legend(('Validtion Accuracy','Training Accuracy'))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    
def gradient_check():
    train_data, valid_data, test_data = load_data()
    model = network2.Network([784,25,10])#([784, 20, 10])
    model.gradient_check(training_data=train_data, layer_id=1, unit_id=5, weight_id=3)

def main():
    # load train_data, valid_data, test_data
    train_data, valid_data, test_data = load_data()
    # construct the network
    model = network2.Network([784,32,10])#([784, 20, 10]) 
    
    # train the network using SGD
    epoch=80
    cost_params=model.SGD(
        training_data=train_data,
        epochs=epoch,
        mini_batch_size=128,
        eta=0.75e-2,
        lmbda = 0.0,
        evaluation_data=valid_data,
        monitor_evaluation_cost=True,
        monitor_evaluation_accuracy=True,
        monitor_training_cost=True,
        monitor_training_accuracy=True)
    print("======Training Complete======")
    
    plot_learning_curve(cost_params,epoch,len(train_data[1]),len(valid_data[1]))
    
    print("======Start  of  Testing======")
    a,yhat=test_data
   
    g=[]
    for i in range(len(a)):
        g.append(model.feedforward(a[i]))
    results = [(np.argmax(x), y)
                        for (x, y) in zip(g,yhat)]
    print("Testing Accuracy is:",sum(int(x == y) for (x, y) in results)/100.0)     
    print("====== End  of  Testing======")
    model.save("Layer3_784_32_10")
    print("Model Saved Succefully")
    print()
    

if __name__ == '__main__':
    FLAGS = get_args()
    if FLAGS.input:
        load_data()
    if FLAGS.sigmoid:
        test_sigmoid()
    if FLAGS.train:
        main()
    if FLAGS.gradient:
        gradient_check()
        
    