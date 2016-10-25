# ELEN4002 Project 16P28, Group 16G18
# Signal Processing of EEG raw data
# author: Danielle Winter 563795
# date: September/ October 2016

import numpy as np
import csv
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score, KFold
from sklearn import metrics
from scipy.stats import sem
from scipy import signal
from sklearn.externals import joblib

from collections import defaultdict
right_columns = defaultdict(list)


# obtain right arm data channels
with open('DaniW-right arm data-18.10.16.13.29.01.csv') as csv_file_right:
    raw_eeg_R = csv.reader(csv_file_right, delimiter=',')
    raw_eeg_R.next()
    for row in raw_eeg_R:
        for (i, v) in enumerate(row):
            right_columns[i].append(v)

af3_right_arm = right_columns[2]
f3_right_arm = right_columns[4]
f4_right_arm = right_columns[13]
af4_right_arm = right_columns[15]


# Obtain lef arm data channels
left_columns = defaultdict(list)

with open('EEG_left_data.csv') as csv_file_left:
    raw_eeg_L = csv.reader(csv_file_left, delimiter=',')
    raw_eeg_L.next()
    for row in raw_eeg_L:
        for (i, v) in enumerate(row):
            left_columns[i].append(v)


f3_left_arm = left_columns[0]
af3_left_arm = left_columns[1]
f4_left_arm = left_columns[2]
af4_left_arm = left_columns[3]

columns = defaultdict(list)
with open('EEG_use_data.csv') as csv_file:
    raw_eeg = csv.reader(csv_file, delimiter=',')
    raw_eeg.next()
    for row in raw_eeg:
        for (i, v) in enumerate(row):
            columns[i].append(v)

F3 = columns[0]         # left side motion channel
F4 = columns[2]        # right side motion channel
AF3 = columns[1]
AF4 = columns[3]

def convert_to_float(s):
    try:
        s=float(s)
    except ValueError:
        pass
    return s


AF3_right = [convert_to_float(s) for s in af3_right_arm]
F3_right = [convert_to_float(s) for s in f3_right_arm]
AF4_right = [convert_to_float(s) for s in af4_right_arm]
F4_right = [convert_to_float(s) for s in f4_right_arm]
AF3_left = [convert_to_float(s) for s in af3_left_arm]
F3_left = [convert_to_float(s) for s in f3_left_arm]
AF4_left = [convert_to_float(s) for s in af4_left_arm]
F4_left = [convert_to_float(s) for s in f4_left_arm]

F3 = [convert_to_float(s) for s in F3]
F4 = [convert_to_float(s) for s in F4]
AF3 = [convert_to_float(s) for s in AF3]
AF4 = [convert_to_float(s) for s in AF4]

seconds = 128*4
dc_level = 4200


def high_pass_filter(data):
        b, a = signal.butter(5, 0.16, 'high', analog=False, output='ba')
        filtered_data = signal.filtfilt(b, a, data, padtype='odd', method='pad')
        return filtered_data


def pop_extra_values(array):
    while len(array) % seconds > 0:
        array.pop()
    return array


def interval_period(array):
    array = high_pass_filter(array)
    val = len(array) / seconds
    array = np.reshape(array, (val, seconds))
    # array = np.array(array) - dc_level
    return array


F3_right_arm = pop_extra_values(F3_right)
F3_right_arm_intervals = interval_period(F3_right_arm)
AF3_right_arm = pop_extra_values(AF3_right)
AF3_right_arm_intervals = interval_period(AF3_right_arm)
F4_right_arm = pop_extra_values(F4_right)
F4_right_arm_intervals = interval_period(F4_right_arm)
AF4_right_arm = pop_extra_values(AF4_right)
AF4_right_arm_intervals = interval_period(AF4_right_arm)

F3_left_arm = pop_extra_values(F3_left)
F3_left_arm_intervals = interval_period(F3_left_arm)
AF3_left_arm = pop_extra_values(AF3_left)
AF3_left_arm_intervals = interval_period(AF3_left_arm)
F4_left_arm = pop_extra_values(F4_left)
F4_left_arm_intervals = interval_period(F4_left_arm)
AF4_left_arm = pop_extra_values(AF4_left)
AF4_left_arm_intervals = interval_period(AF4_left_arm)

F3 = pop_extra_values(F3)
F3_intervals = interval_period(F3)
AF3 = pop_extra_values(AF3)
AF3_intervals = interval_period(AF3)
F4 = pop_extra_values(F4)
F4_intervals = interval_period(F4)
AF4 = pop_extra_values(AF4)
AF4_intervals = interval_period(AF4)

# Split right arm data into stationary and in motion
stationary_data_F3 = [['still', F3_right_arm_intervals[3]], ['still', F3_right_arm_intervals[4]], ['still', F3_right_arm_intervals[5]], ['still', F3_right_arm_intervals[6]], ['still', F3_right_arm_intervals[7]], ['still', F3_right_arm_intervals[8]], ['still', F3_right_arm_intervals[9]], ['still', F3_right_arm_intervals[10]], ['still', F3_right_arm_intervals[11]], ['still', F3_right_arm_intervals[12]], ['still', F3_right_arm_intervals[13]], ['still', F3_right_arm_intervals[14]], ['still', F3_right_arm_intervals[15]], ['still', F3_right_arm_intervals[16]], ['still', F3_right_arm_intervals[17]], ['still', F3_right_arm_intervals[18]], ['still', F3_right_arm_intervals[19]], ['still', F3_right_arm_intervals[20]], ['still', F3_right_arm_intervals[21]], ['still', F3_right_arm_intervals[22]], ['still', F3_right_arm_intervals[23]], ['still', F3_right_arm_intervals[24]]]
stationary_data_AF3 = [['still', AF3_right_arm_intervals[3]], ['still', AF3_right_arm_intervals[4]], ['still', AF3_right_arm_intervals[5]], ['still', AF3_right_arm_intervals[6]], ['still', AF3_right_arm_intervals[7]], ['still', AF3_right_arm_intervals[8]], ['still', AF3_right_arm_intervals[9]], ['still', AF3_right_arm_intervals[10]], ['still', AF3_right_arm_intervals[11]], ['still', AF3_right_arm_intervals[12]], ['still', AF3_right_arm_intervals[13]], ['still', AF3_right_arm_intervals[14]], ['still', AF3_right_arm_intervals[15]], ['still', AF3_right_arm_intervals[16]], ['still', AF3_right_arm_intervals[17]], ['still', AF3_right_arm_intervals[18]], ['still', AF3_right_arm_intervals[19]], ['still', AF3_right_arm_intervals[20]], ['still', AF3_right_arm_intervals[21]], ['still', AF3_right_arm_intervals[22]], ['still', AF3_right_arm_intervals[23]], ['still', AF3_right_arm_intervals[24]]]
stationary_data_F4 = [['still', F4_right_arm_intervals[3]], ['still', F4_right_arm_intervals[4]], ['still', F4_right_arm_intervals[5]], ['still', F4_right_arm_intervals[6]], ['still', F4_right_arm_intervals[7]], ['still', F4_right_arm_intervals[8]], ['still', F4_right_arm_intervals[9]], ['still', F4_right_arm_intervals[10]], ['still', F4_right_arm_intervals[11]], ['still', F4_right_arm_intervals[12]], ['still', F4_right_arm_intervals[13]], ['still', F4_right_arm_intervals[14]], ['still', F4_right_arm_intervals[15]], ['still', F4_right_arm_intervals[16]], ['still', F4_right_arm_intervals[17]], ['still', F4_right_arm_intervals[18]], ['still', F4_right_arm_intervals[19]], ['still', F4_right_arm_intervals[20]], ['still', F4_right_arm_intervals[21]], ['still', F4_right_arm_intervals[22]], ['still', F4_right_arm_intervals[23]], ['still', F4_right_arm_intervals[24]]]
stationary_data_AF4 = [['still', AF4_right_arm_intervals[3]], ['still', AF4_right_arm_intervals[4]], ['still', AF4_right_arm_intervals[5]], ['still', AF4_right_arm_intervals[6]], ['still', AF4_right_arm_intervals[7]], ['still', AF4_right_arm_intervals[8]], ['still', AF4_right_arm_intervals[9]], ['still', AF4_right_arm_intervals[10]], ['still', AF4_right_arm_intervals[11]], ['still', AF4_right_arm_intervals[12]], ['still', AF4_right_arm_intervals[13]], ['still', AF4_right_arm_intervals[14]], ['still', AF4_right_arm_intervals[15]], ['still', AF4_right_arm_intervals[16]], ['still', AF4_right_arm_intervals[17]], ['still', AF4_right_arm_intervals[18]], ['still', AF4_right_arm_intervals[19]], ['still', AF4_right_arm_intervals[20]], ['still', AF4_right_arm_intervals[21]], ['still', AF4_right_arm_intervals[22]], ['still', AF4_right_arm_intervals[23]], ['still', AF4_right_arm_intervals[24]]]

right_arm_move_F3 = [['move RA', F3_right_arm_intervals[42]], ['move RA', F3_right_arm_intervals[44]], ['move RA', F3_right_arm_intervals[46]], ['move RA', F3_right_arm_intervals[49]], ['move RA', F3_right_arm_intervals[51]], ['move RA', F3_right_arm_intervals[57]], ['move RA', F3_right_arm_intervals[62]], ['move RA', F3_right_arm_intervals[66]], ['move RA', F3_right_arm_intervals[80]]]
right_arm_move_AF3 = [['move RA', AF3_right_arm_intervals[42]], ['move RA', AF3_right_arm_intervals[44]], ['move RA', AF3_right_arm_intervals[46]], ['move RA', AF3_right_arm_intervals[49]], ['move RA', AF3_right_arm_intervals[51]], ['move RA', AF3_right_arm_intervals[57]], ['move RA', AF3_right_arm_intervals[62]], ['move RA', AF3_right_arm_intervals[66]], ['move RA', AF3_right_arm_intervals[80]]]
right_arm_move_F4 = [['move RA', F4_right_arm_intervals[42]], ['move RA', F4_right_arm_intervals[44]], ['move RA', F4_right_arm_intervals[46]], ['move RA', F4_right_arm_intervals[49]], ['move RA', F4_right_arm_intervals[51]], ['move RA', F4_right_arm_intervals[57]], ['move RA', F4_right_arm_intervals[62]], ['move RA', F4_right_arm_intervals[66]], ['move RA', F4_right_arm_intervals[80]]]
right_arm_move_AF4 = [['move RA', AF4_right_arm_intervals[42]], ['move RA', AF4_right_arm_intervals[44]], ['move RA', AF4_right_arm_intervals[46]], ['move RA', AF4_right_arm_intervals[49]], ['move RA', AF4_right_arm_intervals[51]], ['move RA', AF4_right_arm_intervals[57]], ['move RA', AF4_right_arm_intervals[62]], ['move RA', AF4_right_arm_intervals[66]], ['move RA', AF4_right_arm_intervals[80]]]

left_arm_move_F3 = [['move LA', F3_left_arm_intervals[3]], ['move LA', F3_left_arm_intervals[6]], ['move LA', F3_left_arm_intervals[10]], ['move LA', F3_left_arm_intervals[15]], ['move LA', F3_left_arm_intervals[21]], ['move LA', F3_left_arm_intervals[24]], ['move LA', F3_left_arm_intervals[28]] ]
left_arm_move_AF3 = [['move LA', AF3_left_arm_intervals[3]], ['move LA', AF3_left_arm_intervals[6]], ['move LA', AF3_left_arm_intervals[10]], ['move LA', AF3_left_arm_intervals[15]], ['move LA', AF3_left_arm_intervals[21]], ['move LA', AF3_left_arm_intervals[24]], ['move LA', AF3_left_arm_intervals[28]] ]
left_arm_move_F4 = [['move LA', F4_left_arm_intervals[3]], ['move LA', F4_left_arm_intervals[6]], ['move LA', F4_left_arm_intervals[10]], ['move LA', F4_left_arm_intervals[15]], ['move LA', F4_left_arm_intervals[21]], ['move LA', F4_left_arm_intervals[24]], ['move LA', F4_left_arm_intervals[28]] ]
left_arm_move_AF4 = [['move LA', AF4_left_arm_intervals[3]], ['move LA', AF4_left_arm_intervals[6]], ['move LA', AF4_left_arm_intervals[10]], ['move LA', AF4_left_arm_intervals[15]], ['move LA', AF4_left_arm_intervals[21]], ['move LA', AF4_left_arm_intervals[24]], ['move LA', AF4_left_arm_intervals[28]] ]

eeg_still_data_F3 = [('still',F3_intervals[1]), ('still',F3_intervals[2]), ('still',F3_intervals[3]), ('still',F3_intervals[4]), ('still',F3_intervals[5]), ('still',F3_intervals[6]), ('still',F3_intervals[7]), ('still',F3_intervals[42], ('still',F3_intervals[43]), 'still',F3_intervals[44]), ('still',F3_intervals[45], ('still',F3_intervals[49]), ('still',F3_intervals[50]), ('still',F3_intervals[51]), ('still',F3_intervals[52])) ]
eeg_moving_data_F3 = [('move', F3_intervals[85]), ('move', F3_intervals[89]), ('move', F3_intervals[100]), ('move', F3_intervals[101]), ('move', F3_intervals[102]), ('move', F3_intervals[103]), ('move', F3_intervals[214]), ('move', F3_intervals[215]), ('move', F3_intervals[216]), ('move', F3_intervals[217]), ('move', F3_intervals[218]), ('move', F3_intervals[219]), ('move', F3_intervals[220]), ('move', F3_intervals[221]), ('move', F3_intervals[222]), ('move', F3_intervals[223])]
eeg_still_data_F4 = [('still',F4_intervals[1]), ('still',F4_intervals[2]), ('still',F4_intervals[3]), ('still',F4_intervals[4]), ('still',F4_intervals[5]), ('still',F4_intervals[6]), ('still',F4_intervals[7]), ('still',F4_intervals[42], ('still',F4_intervals[43]), 'still',F4_intervals[44]), ('still',F4_intervals[45], ('still',F4_intervals[49]), ('still',F4_intervals[50]), ('still',F4_intervals[51]), ('still',F4_intervals[52])) ]
eeg_moving_data_F4 = [('move', F4_intervals[85]), ('move', F4_intervals[89]), ('move', F4_intervals[100]), ('move', F4_intervals[101]), ('move', F4_intervals[102]), ('move', F4_intervals[103]), ('move', F4_intervals[214]), ('move', F4_intervals[215]), ('move', F4_intervals[216]), ('move', F4_intervals[217]), ('move', F4_intervals[218]), ('move', F4_intervals[219]), ('move', F4_intervals[220]), ('move', F4_intervals[221]), ('move', F4_intervals[222]), ('move', F4_intervals[223])]
eeg_still_data_AF3 = [('still',AF3_intervals[1]), ('still',AF3_intervals[2]), ('still',AF3_intervals[3]), ('still',AF3_intervals[4]), ('still',AF3_intervals[5]), ('still',AF3_intervals[6]), ('still',AF3_intervals[7]), ('still',AF3_intervals[42], ('still',AF3_intervals[43]), 'still',AF3_intervals[44]), ('still',AF3_intervals[45], ('still',AF3_intervals[49]), ('still',AF3_intervals[50]), ('still',AF3_intervals[51]), ('still',AF3_intervals[52])) ]
eeg_moving_data_AF3 = [('move', AF3_intervals[85]), ('move', AF3_intervals[89]), ('move', AF3_intervals[100]), ('move', AF3_intervals[101]), ('move', AF3_intervals[102]), ('move', AF3_intervals[103]), ('move', AF3_intervals[214]), ('move', AF3_intervals[215]), ('move', AF3_intervals[216]), ('move', AF3_intervals[217]), ('move', AF3_intervals[218]), ('move', AF3_intervals[219]), ('move', AF3_intervals[220]), ('move', AF3_intervals[221]), ('move', AF3_intervals[222]), ('move', AF3_intervals[223])]
eeg_still_data_AF4 = [('still',AF4_intervals[1]), ('still',AF4_intervals[2]), ('still',AF4_intervals[3]), ('still',AF4_intervals[4]), ('still',AF4_intervals[5]), ('still',AF4_intervals[6]), ('still',AF4_intervals[7]), ('still',AF4_intervals[42], ('still',AF4_intervals[43]), 'still',AF4_intervals[44]), ('still',AF4_intervals[45], ('still',AF4_intervals[49]), ('still',AF4_intervals[50]), ('still',AF4_intervals[51]), ('still',AF4_intervals[52])) ]
eeg_moving_data_AF4 = [('move', AF4_intervals[85]), ('move', AF4_intervals[89]), ('move', AF4_intervals[100]), ('move', AF4_intervals[101]), ('move', AF4_intervals[102]), ('move', AF4_intervals[103]), ('move', AF4_intervals[214]), ('move', AF4_intervals[215]), ('move', AF4_intervals[216]), ('move', AF4_intervals[217]), ('move', AF4_intervals[218]), ('move', AF4_intervals[219]), ('move', AF4_intervals[220]), ('move', AF4_intervals[221]), ('move', AF4_intervals[222]), ('move', AF4_intervals[223])]


def data_selection(data):
    output_data = [tuple_item[1] for tuple_item in data]
    return output_data


def class_selection(data):
    output_class = [tuple_item[0] for tuple_item in data]
    return output_class


F3_database = eeg_still_data_F3 + stationary_data_F3 + eeg_moving_data_F3 + right_arm_move_F3 + left_arm_move_F3
AF3_database = eeg_still_data_AF3 + stationary_data_AF3 + eeg_moving_data_AF3 + right_arm_move_AF3 + left_arm_move_AF3
F4_database = eeg_still_data_F4 + stationary_data_F4 + eeg_moving_data_F4 + right_arm_move_F4 + left_arm_move_F4
AF4_database = eeg_still_data_AF4 + stationary_data_AF4 + eeg_moving_data_AF4 + right_arm_move_AF4 + left_arm_move_AF4

F3_data = data_selection(F3_database)
AF3_data = data_selection(AF3_database)
F4_data = data_selection(F4_database)
AF4_data = data_selection(AF4_database)

F3_class = class_selection(F3_database)
AF3_class = class_selection(AF3_database)
F4_class = class_selection(F4_database)
AF4_class = class_selection(AF4_database)

# Machine Learning - Training of SVM
F3_classifier = SVC(kernel='linear')
AF3_classifier = SVC(kernel='linear')
F4_classifier = SVC(kernel='linear')
AF4_classifier = SVC(kernel='linear')

X_train_F3, X_test_F3, Y_train_F3, Y_test_F3 = train_test_split(F3_data, F3_class, test_size = 0.25, random_state= 0)
X_train_AF3, X_test_AF3, Y_train_AF3, Y_test_AF3 = train_test_split(AF3_data, AF3_class, test_size = 0.25, random_state= 0)
X_train_F4, X_test_F4, Y_train_F4, Y_test_F4 = train_test_split(F4_data, F4_class, test_size = 0.25, random_state= 0)
X_train_AF4, X_test_AF4, Y_train_AF4, Y_test_AF4 = train_test_split(AF4_data, AF4_class, test_size = 0.25, random_state= 0)


# Calculate a cross-validation score
def evaluate_cross_validation(classifier, X, y, K):
    cross_val = KFold(len(y), K, shuffle=True, random_state=0)
    scores = cross_val_score(classifier, X, y, cv=cross_val)
    print ('Mean score: {0:.3f}(+/-{0:.3f})').format(np.mean(scores), sem(scores))


# Evaluates the accuracy of the classifier
def train_and_evaluate(classifier, x_train, x_test, y_train, y_test):
    classifier.fit(x_train,y_train)
    print "Accuracy on training set: "
    print classifier.score(x_train,y_train)
    print "Accuracy on testing set: "
    print classifier.score(x_test, y_test)
    predict_y = classifier.predict(x_test)
    print predict_y
    print "Classification report: "
    print metrics.classification_report(y_test, predict_y)


# cross validate and fit data
print "F3 evaluation and training"
evaluate_cross_validation(F3_classifier, X_train_F3, Y_train_F3, 5)
train_and_evaluate(F3_classifier, X_train_F3, X_test_F3, Y_train_F3, Y_test_F3)

print "AF3 evaluation and training"
evaluate_cross_validation(AF3_classifier, X_train_AF3, Y_train_AF3, 5)
train_and_evaluate(AF3_classifier, X_train_AF3, X_test_AF3, Y_train_AF3, Y_test_AF3)

print "F4 evaluation and training"
evaluate_cross_validation(F4_classifier, X_train_F4, Y_train_F4, 5)
train_and_evaluate(F4_classifier, X_train_F4, X_test_F4, Y_train_F4, Y_test_F4)

print "AF4 evaluation and training"
evaluate_cross_validation(AF4_classifier, X_train_AF4, Y_train_AF4, 5)
train_and_evaluate(AF4_classifier, X_train_AF4, X_test_AF4, Y_train_AF4, Y_test_AF4)


# Fit all data
F3_classifier.fit(F3_data, F3_class)
F4_classifier.fit(F4_data, F4_class)
AF3_classifier.fit(AF3_data, AF3_class)
AF4_classifier.fit(AF4_data, AF4_class)

# Serialise trained SVC objects
joblib.dump(F3_classifier, 'F3_classifier.pkl')
joblib.dump(AF3_classifier, 'AF3_classifier.pkl')
joblib.dump(F4_classifier, 'F4_classifier.pkl')
joblib.dump(AF4_classifier, 'AF4_classifier.pkl')
