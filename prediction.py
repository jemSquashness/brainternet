# ELEN4002 Project 16P28, Group 16G18
# Signal Processing of EEG raw data
# author: Danielle Winter 563795
# date: September/ October 2016

import numpy as np
from sklearn.svm import SVC
from sklearn.externals import joblib

F3_classifier = joblib.load('F3_classifier.pkl')
AF3_classifier = joblib.load('AF3_classifier.pkl')
F4_classifier = joblib.load('F4_classifier.pkl')
AF4_classifier = joblib.load('F4_classifier.pkl')


def prediction_on_new_data(classifier, new_data):
    predicted_input = classifier.predict(new_data)
    print predicted_input

