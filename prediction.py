# ELEN4002 Project 16P28, Group 16G18
# Signal Processing of EEG raw data
# author: Danielle Winter 563795
# date: September/ October 2016

import numpy as np
from sklearn.svm import SVC
from sklearn.externals import joblib
from scipy import signal

F3_classifier = joblib.load('F3_classifier.pkl')
AF3_classifier = joblib.load('AF3_classifier.pkl')
F4_classifier = joblib.load('F4_classifier.pkl')
AF4_classifier = joblib.load('F4_classifier.pkl')


def prediction_on_new_data(classifier, new_data):
    predicted_input = classifier.predict(new_data)
    print predicted_input


def high_pass_filter(data):
    b, a = signal.butter(5, 0.16, 'high', analog=False, output='ba')
    filtered_data = signal.filtfilt(b, a, data, padtype='odd', method='pad')
    return filtered_data


# Read in new data

# Filter new data
F3_filt = high_pass_filter(F3_4s_data)
AF3_filt = high_pass_filter(AF3_4s_data)
F4_filt = high_pass_filter(F4_4s_data)
AF4_filt = high_pass_filter(AF4_4s_data)

F3_predicted_input = prediction_on_new_data(F3_classifier, F3_filt)
AF3_predicted_input = prediction_on_new_data(AF3_classifier, AF3_filt)
F4_predicted_input = prediction_on_new_data(F4_classifier, F4_filt)
AF4_predicted_input = prediction_on_new_data(AF4_classifier, AF4_filt)