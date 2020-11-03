import os
import sys
import math
import numpy as np

def l2(v1, v2):
    summation = 0
    for i in range(len(v1) - 1):
        summation += ((v1[i] - v2[i])**2)
    summation *= 0.5
    return summation

def mean(array):
    return sum(array) / len(array)

def stdev(array, mean):
    summation = 0
    for num in array:
        summation += ((num - mean)**2)
    summation /= (len(array) - 1)
    summation = math.sqrt(summation)
    if(summation == 0):
        return 1
    else:
        return summation

def knn_classify():
    if len(sys.argv) < 4:
        print('Invalid Number of Args')
        exit()

    training_path = sys.argv[1]
    testing_path =  sys.argv[2]
    k = int(sys.argv[3])

    if not os.path.exists(training_path):
        print('Invalid File Training File Path')
        exit()
    training_file = open(training_path)
    training_lines = training_file.readlines()
    if not os.path.exists(testing_path):
        print('Invalid File Testing File Path')
        exit()
    testing_file = open(testing_path)
    testing_lines = testing_file.readlines()

    training_data = []
    testing_data = []

    for line in training_lines:
        line = line.split()
        for i in range(len(line)):
            line[i] = float(line[i])
        training_data.append(line)

    for line in testing_lines:
        line = line.split()
        for i in range(len(line)):
            line[i] = float(line[i])
        testing_data.append(line)
    
    vals = []

    for i in range(len(training_data[0]) - 1):
        attribute = []
        for j in range(len(training_data)):
            attribute.append(training_data[j][i])
        m = mean(attribute)
        s = stdev(attribute, m)
        vals.append((m, s))
        for l in range(len(training_data)):
            training_data[l][i] = (training_data[l][i] - m) / s
    
    for i in range(len(testing_data[0]) - 1):
        for j in range(len(testing_data)):
            testing_data[j][i] = (testing_data[j][i] - vals[i][0]) / vals[i][1]
    
    object_id = 1
    total_correct = 0
    
    for i in range(len(testing_data)):
        distances = []
        accuracy = 0
        ties = 0
        for j in range(len(training_data)):
            distances.append((l2(testing_data[i], training_data[j]), training_data[j][-1]))
        distances = sorted(distances)
        prediction = {}
        for l in range(k):
            if distances[l][1] not in prediction:
                prediction[distances[l][1]] = 1
            else: 
                prediction[distances[l][1]] += 1
        finals = sorted(prediction.items(), key = lambda x: x[1], reverse = True)
        best_class = finals[0][0]
        best_val = finals[0][1]
        for num in finals:
            if num[1] == best_val:
                if num[0] == testing_data[i][-1]:
                    accuracy = 1
                ties += 1
            else:
                break
        accuracy /= ties
        total_correct += accuracy
        print('ID=%5d, predicted=%3d, true=%3d, accuracy=%4.2f' % (object_id, best_class, testing_data[i][-1], accuracy))
        object_id += 1
    
    print('classification accuracy=%6.4f\n' % (total_correct / len(testing_data)))

knn_classify()