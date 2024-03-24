# -*- coding: utf-8 -*-
"""KNN_DISHACHIPLONKER.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/18tNzSeQr3bxb0QUYh8ydYQdLwF74MBQd

## **Q1.1 and Q1.2**

The below code and it's output show the graph for the accuracies of the training and testing data. 
The first graph is a combined graph for training and testing data showing the accuracy on the vertical axis and values of k on the horizontal axis.
The second graph is a graph showing the value of k on the horizontal axis,
and on the vertical axis, the average accuracy of models trained over the training set, given that particular value of k, for each point,the corresponding standard deviation is shown.
The third graph is a graph showing the value of k on the horizontal axis,
and on the vertical axis, the average accuracy of models trained over the testing set, given that particular value of k, for each point,the corresponding standard deviation is shown.
"""

import pandas as pd 
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from operator import itemgetter


#Function to predict the class label of a test row using knn
def classification_prediction(train, train_labels, test_row, k):
  neighbours = find_neighbours(train, train_labels, test_row, k)

  #get the class labels of the k nearest neighbours
  output_values = []

  for neighbour in neighbours :
    output_values.append(neighbour[-1])

  #find most common neighbours class label among k neighbours
  count={}
  for v in output_values:
    if v in count:
      count[v] += 1
    else:
      count[v] = 1

  prediction = max(count, key=count.get)

  return prediction

#Function to find the nearest neighbours
def find_neighbours(train, train_labels, test_row, k):
  distances = []

  for d in range(len(train)):
    distance = eud_dist(test_row, train[d])

    #Append euclidian distance along with corresponding rw and lable to distance list
    distances.append((train[d], train_labels[d], distance))

  #Sort the distances list in the ascending order based on the distance
  distances.sort(key=itemgetter(2))

  neighbours = []
  for n in range(k):
    neighbours.append((distances[n][0], distances[n][1]))

  return neighbours

#Function implements knn for all the rows in train/test data
def k_n_neighbours_implement(train, train_labels, test, k):
  prediction = []

  #Loop through each row in test set and predict the label for it 
  for rowno in test:
    output = classification_prediction(train, train_labels, rowno, k)
    prediction.append(output)

  return prediction

#Function to find euclidian distance between two data points
def eud_dist(p1, p2):
  calcdist = 0.0
  for d in range(len(p1)):
    difference = p1[d] - p2[d]

    calcdist += difference ** 2

  return math.sqrt(calcdist)

#Load dataset
#from google.colab import files
#uploaded = files.upload()

datafile = pd.read_csv("iris.csv", header=None)
#datafile = shuffle(datafile, random_state=42)

X = datafile.iloc[:, :-1].values
y = datafile.iloc[:, -1].values

X, y = shuffle(X, y, random_state=36)

#Test the kNN algorithm
k_vals = range(1, 52, 2)

accurancies_train = []
accurancies_test = []

stdevs_train = []
stdevs_test = []

for k in k_vals:

  train_scores = []
  test_scores = []

  i = 0
  while i < 20:

    #Split dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=36, shuffle=True)

    #Normalize data
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    #Predicting for test and train data
    pred_y_train = k_n_neighbours_implement(X_train, y_train, X_train, k)
    pred_y_test = k_n_neighbours_implement(X_train, y_train, X_test, k)

    #calculate accuracy for test and train data
    train_accuracylist = sum(y_train == pred_y_train) / len(y_train)
    test_accuracylist = sum(y_test == pred_y_test) / len(y_test)

    #appending accuracies for all 20 iterations
    train_scores.append(train_accuracylist)
    test_scores.append(test_accuracylist)

    i += 1

  train_mean = np.mean(np.array(train_scores),0)
  test_mean = np.mean(np.array(test_scores),0)

  train_stddev = np.std(np.array(train_scores),0, ddof=1)
  test_stddev = np.std(np.array(test_scores),0, ddof=1)

  accurancies_train.append(train_mean)
  accurancies_test.append(test_mean)

  stdevs_train.append(train_stddev)
  stdevs_test.append(test_stddev)


plt.plot(k_vals, accurancies_train, label='Training accuracy')
plt.plot(k_vals, accurancies_test, label='Testing accuracy')
plt.ylabel('Accuracies')
plt.xlabel('k')
plt.title('Accuracy and k')
plt.xticks(k_vals)
plt.legend()
plt.show()

#Converting std devs lists to array
train_stdevs_arr = np.array(stdevs_train)
test_stdevs_arr = np.array(stdevs_test)

#Train Data Plots
plt.errorbar(k_vals, accurancies_train, xerr=train_stdevs_arr, label='Training set', capsize=15)
plt.plot(k_vals, accurancies_train)
plt.ylabel('Accuracies')
plt.xlabel('k')
plt.title('Train Set Accuracy')
plt.xticks(k_vals)
plt.show()

#Tesrt Data Plots
plt.errorbar(k_vals, accurancies_test, xerr=test_stdevs_arr, label='Testing set', capsize=15)
plt.plot(k_vals, accurancies_test)
plt.ylabel('Accuracies')
plt.xlabel('k')
plt.title('Test Set Accuracy')
plt.xticks(k_vals)
plt.show()

# plt.errorbar(k_vals, accurancies_test, xerr=None, yerr=test_stddev)
# plt.plot(k_vals, accurancies_test)
# plt.title('Testing Set accuracy')
# plt.xlabel('k value')
# plt.ylabel('Accuracy percentage')
# plt.show()

"""## **Q1.3**

The accuracy on the training dataset decreases as k increases since the model becomes less complex. Misclassification can occur for higher values of k since the model starts considering more neighbours.The training accuracy starts from 1 because model is perfectly fit on training set when k=1.

Instead the accuracy of the testing dataset is good as k increases and then starts to decrease. Small value of k can lead to overfitting. Model becomes too general and classifies poorly.

## **Q1.4**

The model is underfitting if training accuracy is low and testing accuracy is also low. Model is not complex enough to capture underlying data in this case.
It usually occurs when k is large as well. In this plot, k would be underfitting for a range over k>25.

The model is overfitting if training accuracy is high and testing accuracy is low. Model is probably fitting to noise in the data here. It occurs when k is small as well. In this plot, k would be overfitting for a range over k<10.

## **Q1.5**

Based on plot generated, testing accuracy is highest for k = 11, after that testing accuracy decreases. Choosing k=11 would provide a good balance between overfitting or underfitting. Therefore, I would choose k=11 to fine-tune this algorithm so that it works well in real life as well.

## **Q1.6**

# a)
"""

#WITHOUT NORMALIZATION

import pandas as pd 
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from operator import itemgetter


#Function to predict the class label of a test row using knn
def classification_prediction(train, train_labels, test_row, k):
  neighbours = find_neighbours(train, train_labels, test_row, k)

  #get the class labels of the k nearest neighbours
  output_values = []

  for neighbour in neighbours :
    output_values.append(neighbour[-1])

  #find most common neighbours class label among k neighbours
  count={}
  for v in output_values:
    if v in count:
      count[v] += 1
    else:
      count[v] = 1

  prediction = max(count, key=count.get)

  return prediction

#Function to find the nearest neighbours
def find_neighbours(train, train_labels, test_row, k):
  distances = []

  for d in range(len(train)):
    distance = eud_dist(test_row, train[d])

    #Append euclidian distance along with corresponding rw and lable to distance list
    distances.append((train[d], train_labels[d], distance))

  #Sort the distances list in the ascending order based on the distance
  distances.sort(key=itemgetter(2))

  neighbours = []
  for n in range(k):
    neighbours.append((distances[n][0], distances[n][1]))

  return neighbours

#Function implements knn for all the rows in train/test data
def k_n_neighbours_implement(train, train_labels, test, k):
  prediction = []

  #Loop through each row in test set and predict the label for it 
  for rowno in test:
    output = classification_prediction(train, train_labels, rowno, k)
    prediction.append(output)

  return prediction

#Function to find euclidian distance between two data points
def eud_dist(p1, p2):
  calcdist = 0.0
  for d in range(len(p1)):
    difference = p1[d] - p2[d]

    calcdist += difference ** 2

  return math.sqrt(calcdist)

#Load dataset
#from google.colab import files
#uploaded = files.upload()

datafile = pd.read_csv("iris.csv", header=None)
#datafile = shuffle(datafile, random_state=42)

X = datafile.iloc[:, :-1].values
y = datafile.iloc[:, -1].values

X, y = shuffle(X, y, random_state=36)

#Test the kNN algorithm
k_vals = range(1, 52, 2)

accurancies_train = []
accurancies_test = []

stdevs_train = []
stdevs_test = []

for k in k_vals:

  train_scores = []
  test_scores = []

  i = 0
  while i < 20:

    #Split dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=36, shuffle=True)

    #Normalize data
    #scaler = MinMaxScaler()
    #X_train = scaler.fit_transform(X_train)
    #X_test = scaler.transform(X_test)

    #Predicting for test and train data
    pred_y_train = k_n_neighbours_implement(X_train, y_train, X_train, k)
    pred_y_test = k_n_neighbours_implement(X_train, y_train, X_test, k)

    #calculate accuracy for test and train data
    train_accuracylist = sum(y_train == pred_y_train) / len(y_train)
    test_accuracylist = sum(y_test == pred_y_test) / len(y_test)

    #appending accuracies for all 20 iterations
    train_scores.append(train_accuracylist)
    test_scores.append(test_accuracylist)

    i += 1

  train_mean = np.mean(np.array(train_scores),0)
  test_mean = np.mean(np.array(test_scores),0)

  train_stddev = np.std(np.array(train_scores),0, ddof=1)
  test_stddev = np.std(np.array(test_scores),0, ddof=1)

  accurancies_train.append(train_mean)
  accurancies_test.append(test_mean)

  stdevs_train.append(train_stddev)
  stdevs_test.append(test_stddev)


plt.plot(k_vals, accurancies_train, label='Training accuracy')
plt.plot(k_vals, accurancies_test, label='Testing accuracy')
plt.ylabel('Accuracies')
plt.xlabel('k')
plt.title('Accuracy and k')
plt.xticks(k_vals)
plt.legend()
plt.show()

#Converting std devs lists to array
train_stdevs_arr = np.array(stdevs_train)
test_stdevs_arr = np.array(stdevs_test)

#Train Data Plots
plt.errorbar(k_vals, accurancies_train, xerr=train_stdevs_arr, label='Training set', capsize=15)
plt.plot(k_vals, accurancies_train)
plt.ylabel('Accuracies')
plt.xlabel('k')
plt.title('Train Set Accuracy')
plt.xticks(k_vals)
plt.show()

#Tesrt Data Plots
plt.errorbar(k_vals, accurancies_test, xerr=test_stdevs_arr, label='Testing set', capsize=15)
plt.plot(k_vals, accurancies_test)
plt.ylabel('Accuracies')
plt.xlabel('k')
plt.title('Test Set Accuracy')
plt.xticks(k_vals)
plt.show()

"""# b)

The best value of k on the testing set in the above graph is k = 7

# c)

The above version without feature normalization might perform worse than the version with normalization. The algorithm assumes all features to have equal importance in calculating distance between two data points. Between any two features, if one of features has larger scales then that may dominate and mask the smaller scaled features in distance calculation.
Normalization ensures that the distance metric treats all data points equally which results in better performance and correct identification of nearest neighbours.
The algorithm does show better performance for the version with normalization which in the other case, dips or decreases. But the above stated reasons still stand true for either cases.
"""