# ELEN4002 Project 16P28, Group 16G18
# Signal Processing of EEG raw data
# author: Danielle Winter 563795
# date: September/ October 2016

import numpy as np
from sklearn.svm import SVC
from sklearn.externals import joblib
from scipy import signal
import pickle


F3_classifier = pickle.load(open('F3_classifier.pickle'))
AF3_classifier = pickle.load(open('AF3_classifier.pickle'))
F4_classifier = pickle.load(open('F4_classifier.pickle'))
AF4_classifier = pickle.load(open('F4_classifier.pickle'))


def feature_extraction(array):
    for sample in range(len(array)):
        array[sample] = array[sample]*window
        array[sample] = fft(array[sample], 512)
        #array[sample] = abs(array[sample])
    return array


def statistical_features(frequency_data):
    features = []
    for row in range(len(frequency_data)):
        magnitude = abs(frequency_data[row])
        phase = np.angle(frequency_data[row])
        stat_mean = np.mean(magnitude)
        stat_max = np.max(magnitude)
        stat_min = np.min(magnitude)
        mag_var = np.var(magnitude)
        phase_var = np.var(phase)
        stats = [stat_mean, stat_max, stat_min, mag_var, phase_var]
        features = np.append(features, stats)
    features = np.reshape(features, [len(frequency_data), 5])
    return features


def high_pass_filter(data):
    b, a = signal.butter(5, 0.16, 'high', analog=False, output='ba')
    filtered_data = signal.filtfilt(b, a, data, padtype='odd', method='pad')
    return filtered_data


def prediction_on_new_data(classifier, new_data):
    predicted_input = classifier.predict(new_data)
    return predicted_input

# Read in new data
#channelnumber_4s_data is an array of floats of the new eeg data


# Filter new data
F3_filt = high_pass_filter(F3_4s_data)
F3_fft = feature_extraction(F3_filt)
F3_features = statistical_features(F3_fft)
AF3_filt = high_pass_filter(AF3_4s_data)
AF3_fft = feature_extraction(AF3_filt)
AF3_features = statistical_features(AF3_fft)
F4_filt = high_pass_filter(F4_4s_data)
F4_fft = feature_extraction(F4_filt)
F4_features = statistical_features(F4_fft)
AF4_filt = high_pass_filter(AF4_4s_data)
AF4_fft = feature_extraction(AF4_filt)
AF4_features = statistical_features(AF4_fft)


F3_predicted_input = prediction_on_new_data(F3_classifier, F3_features)
AF3_predicted_input = prediction_on_new_data(AF3_classifier, AF3_features)
F4_predicted_input = prediction_on_new_data(F4_classifier, F4_features)
AF4_predicted_input = prediction_on_new_data(AF4_classifier, AF4_features)