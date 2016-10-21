# ELEN4002 Project 16P28, Group 16G18
# Signal Processing of EEG raw data
# author: Danielle Winter 563795
# date: September/ October 2016
import csv
import numpy as np
import sklearn as sk
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score, KFold
from sklearn import metrics
from scipy.stats import sem
from numpy.fft import fft, fftshift
import pandas as pd

from collections import defaultdict

columns = defaultdict(list) # each value in each column is appended to a list

# reading in csv file data
with open('DanielleWinter-TestData06_09-06.09.16.10.15.43.csv') as csv_file:
    raw_eeg = csv.reader(csv_file, delimiter=',')
    raw_eeg.next()
    for row in raw_eeg:
        for (i, v) in enumerate(row):
            columns[i].append(v)


# Setting the channels to the correct electrodes
# AF3 = columns[2]
# F7 = columns[3]
F3 = columns[4]         # left side motion channel
# FC5 = columns[5]
# T7 = columns[6]
# P7 = columns[7]
# O1 = columns[8]
# O2 = columns[9]
# P8 = columns[10]
# T8 = columns[11]
# FC6 = columns[12]
F4 = columns[13]        # right side motion channel
# F8 = columns[14]
# AF4 = columns[15]

def convert_to_float(s):
    try:
        s=float(s)
    except ValueError:
        pass
    return s

F3 = [convert_to_float(s) for s in F3]
F4 = [convert_to_float(s) for s in F4]
#splitting the data into 4 second intervals (due to sampling time, it is a power of 2 for windowing an d fft)
# F3 and F4 are main motor channels
seconds = 128*4

while len(F3) % seconds > 0:
    F3.pop()    # losing a few samples of data - not a problem with live stream

while len(F4) % seconds > 0:
    F4.pop()
b = len(F3)/seconds

F3_intervals = np.reshape(F3,(b, seconds))

# window = np.hanning(seconds)
# F3_win = F3_intervals[0,:]*window # window the interval data before applying fft


while len(F4) % seconds > 0:
    F4.pop()
c = len(F4)/seconds
F4_intervals = np.reshape(F4,(c, seconds))

# Still and motion data segments
eeg_still_data_L = [('still',F3_intervals[1]), ('still',F3_intervals[2]), ('still',F3_intervals[3]), ('still',F3_intervals[4]), ('still',F3_intervals[5]), ('still',F3_intervals[6]), ('still',F3_intervals[7]), ('still',F3_intervals[42], ('still',F3_intervals[43]), 'still',F3_intervals[44]), ('still',F3_intervals[45], ('still',F3_intervals[49]), ('still',F3_intervals[50]), ('still',F3_intervals[51]), ('still',F3_intervals[52])) ]
eeg_moving_data_L = [('move', F3_intervals[85]), ('move', F3_intervals[89]), ('move', F3_intervals[100]), ('move', F3_intervals[101]), ('move', F3_intervals[102]), ('move', F3_intervals[103]), ('move', F3_intervals[214]), ('move', F3_intervals[215]), ('move', F3_intervals[216]), ('move', F3_intervals[217]), ('move', F3_intervals[218]), ('move', F3_intervals[219]), ('move', F3_intervals[220]), ('move', F3_intervals[221]), ('move', F3_intervals[222]), ('move', F3_intervals[223])]

eeg_still_data_R = [('still',F4_intervals[1]), ('still',F4_intervals[2]), ('still',F4_intervals[3]), ('still',F4_intervals[4]), ('still',F4_intervals[5]), ('still',F4_intervals[6]), ('still',F4_intervals[7]), ('still',F4_intervals[42], ('still',F4_intervals[43]), 'still',F4_intervals[44]), ('still',F4_intervals[45], ('still',F4_intervals[49]), ('still',F4_intervals[50]), ('still',F4_intervals[51]), ('still',F4_intervals[52])) ]
eeg_moving_data_R = [('move', F4_intervals[85]), ('move', F4_intervals[89]), ('move', F4_intervals[100]), ('move', F4_intervals[101]), ('move', F4_intervals[102]), ('move', F4_intervals[103]), ('move', F4_intervals[214]), ('move', F4_intervals[215]), ('move', F4_intervals[216]), ('move', F4_intervals[217]), ('move', F4_intervals[218]), ('move', F4_intervals[219]), ('move', F4_intervals[220]), ('move', F4_intervals[221]), ('move', F4_intervals[222]), ('move', F4_intervals[223])]


# Create a database of tuples with class and data
left_eeg_database = eeg_still_data_L + eeg_moving_data_L # filtered eeg data for classification
left_eeg_data = [tuple_item[1] for tuple_item in left_eeg_database]


right_eeg_database = eeg_still_data_R + eeg_moving_data_R # filtered eeg data for classification
right_eeg_data = [tuple_item[1] for tuple_item in right_eeg_database]
# Change data labels to integers for SVC module
# Left and right arm motions are added for now
def label_to_int(label):
    if label == 'still':
        return 1
    elif label == 'move':
        return 2
    elif label == 'right_arm':
        return 3
    elif label == 'left_arm':
        return 4
    else:
        print "Invalid Label"
# return an array with the classes of data
eeg_data_class_L = [tuple_item[0] for tuple_item in left_eeg_database]
eeg_data_class_R = [tuple_item[0] for tuple_item in right_eeg_database]
# motion_data = [] # indices of motion data in eeg_data

# machine learning SVM
# SVM is done separately for motion channels.
# If motion is detected in F3, the individual is moving the
# right hand side of their body which corresponds
# to increased activity in the left hemisphere
# The contrary is true for the F4 channel

svc1 = SVC(kernel='linear')

X_train_L, X_test_L, Y_train_L, Y_test_L = train_test_split(left_eeg_data, eeg_data_class_L, test_size = 0.25, random_state= 0)
X_train_R, X_test_R, Y_train_R, Y_test_R = train_test_split(right_eeg_data, eeg_data_class_R, test_size = 0.25, random_state= 0)
# Split the data into training and testing sets
# The testing portion is 25% of the data
def evaluate_cross_validation(classifier,X, y, K):
    cross_val = KFold(len(y),K, shuffle=True, random_state=0)
    scores = cross_val_score(classifier, X, y, cv=cross_val)
    print ('Mean score: {0:.3f}(+/-{0:.3f})').format(np.mean(scores),sem(scores))

def train_and_evaluate(classifier, x_train, x_test, y_train, y_test):
    classifier.fit(x_train,y_train)
    print "Accuracy on training set: "
    print classifier.score(x_train,y_train)
    print "Accuracy on testing set: "
    print classifier.score(x_test, y_test)
    predict_y = classifier.predict(x_test)
    print "Classification report: "
    print metrics.classification_report(y_test, predict_y)

train_and_evaluate(svc1, X_train_L, X_test_L, Y_train_L, Y_test_L)
train_and_evaluate(svc1, X_train_R, X_test_R, Y_train_R, Y_test_R)



def prediction_on_new_data(classifier, new_data_left, new_data_right):
    predicted_input_L = classifier.predict(new_data_left)
    predicted_input_R = classifier.predict(new_data_right)
    print predicted_input_L
    print predicted_input_R

# Bringing in new eeg data segment
new_data_left = [] # 4 seconds of F3
new_data_right = [] # 4 seconds of F4
prediction_on_new_data(svc1, new_data_left, new_data_right)