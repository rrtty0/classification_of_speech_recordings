import csv
import os
import librosa.display
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier


def normalize_duration(class_, filename, minimum_duration):
    song_name = str(os.getcwd()) + f'\\{class_}\\{filename}'
    y, sr = librosa.load(song_name, sr=22050, mono=True)
    file_duration = librosa.get_duration(y, sr)
    delta = file_duration - minimum_duration
    y_new, sr = librosa.load(song_name, sr=22050, mono=True, offset=delta / 2, duration=minimum_duration)
    return y_new, sr

def create_train_csv():
    header = 'class song_filename chroma_stft rms spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
    for i in range(1, 21):
        header += f' mfcc{i}'
    header = header.split()

    file = open(dataset_train_file, 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)

def create_test_csv():
    header = 'song_filename chroma_stft rms spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
    for i in range(1, 21):
        header += f' mfcc{i}'
    header = header.split()

    file = open(dataset_test_file, 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)

def create_results_csvs():
    header = 'Filename Predict Probability'
    header = header.split()
    file_names = [result_file_dtc, result_file_neigh, result_file_svc, result_file_rfc, result_file_gbc]

    for file_name in file_names:
        file = open(file_name, 'w', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(header)

def train_to_csv():
    print('\nTrain split of songs:')
    for class_ in classes:
        for song_filename in os.listdir(path_to_root + f'{class_}'):
            # y, sr = normalize_duration(class_, song_filename, minimum_duration)
            full_song_name = path_to_root + f'{class_}\\{song_filename}'
            y, sr = librosa.load(full_song_name, sr=22050, mono=True)
            print('Class: ' + class_ + '; Song filename: ' + song_filename + '; Duration: ' + str(librosa.get_duration(y)))
            chroma_stft, rms, spec_cent, spec_bw, rolloff, zcr, mfcc = create_features(y, sr)
            output_train_to_csv(class_, song_filename, chroma_stft, rms, spec_cent, spec_bw, rolloff, zcr, mfcc)
        print('---------------')
    print('Features successful created and outputed at dataset_train.csv')

def test_to_csv():
    print('\nTest split of songs:')
    for song_filename in os.listdir(path_to_root + dataset_test_folder):
        # y, sr = normalize_duration(dataset_test_folder, song_filename, minimum_duration)
        full_song_name = path_to_root + dataset_test_folder + f'\\{song_filename}'
        y, sr = librosa.load(full_song_name, sr=22050, mono=True)
        print('Song filename: ' + song_filename + '; Duration: ' + str(librosa.get_duration(y)))
        chroma_stft, rms, spec_cent, spec_bw, rolloff, zcr, mfcc = create_features(y, sr)
        output_test_to_csv(song_filename, chroma_stft, rms, spec_cent, spec_bw, rolloff, zcr, mfcc)
    print('---------------')
    print('Features successful created and outputed at dataset_test.csv\n')

def create_features(y, sr):
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    return chroma_stft, rms, spec_cent, spec_bw, rolloff, zcr, mfcc

def output_train_to_csv(class_, song_filename, chroma_stft, rms, spec_cent, spec_bw, rolloff, zcr, mfcc):
    to_append = f'{class_} {song_filename} {np.mean(chroma_stft)} {np.mean(rms)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
    for i in mfcc:
        to_append += f' {np.mean(i)}'
    to_append = to_append.split()

    file = open(dataset_train_file, 'a', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(to_append)

def output_test_to_csv(song_filename, chroma_stft, rms, spec_cent, spec_bw, rolloff, zcr, mfcc):
    to_append = f'{song_filename} {np.mean(chroma_stft)} {np.mean(rms)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
    for i in mfcc:
        to_append += f' {np.mean(i)}'
    to_append = to_append.split()

    file = open(dataset_test_file, 'a', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(to_append)

def output_results_to_csvs():
    results = [result_dtc, result_neigh, result_svc, result_rfc, result_gbc]
    file_names = [result_file_dtc, result_file_neigh, result_file_svc, result_file_rfc, result_file_gbc]

    for i in range(len(results)):
        result = results[i]
        result_file = file_names[i]
        for i in range(len(result[0])):
            to_append = f'{result[0][i]} {result[1][i]} {result[2][i]}'
            to_append = to_append.split()
            file = open(result_file, 'a', newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(to_append)

def output_results_to_console():
    results = [result_dtc, result_neigh, result_svc, result_rfc, result_gbc]
    dict = {0:'DecisionTree', 1:'kNeighbors', 2:'SVC', 3:'RandomForest', 4:'GradientBoosting'}

    for i in range(len(results)):
        result = results[i]
        print('\n Result - ' + dict[i] + ':')
        prob_counter = [0, 0, 0, 0, 0]
        sum_probabilities = 0
        for i in range(len(result[0])):
            print('Filename: ' + str(result[0][i]) + '; Predict: ' + str(result[1][i]) + '; Probability: ' + str(
                result[2][i]))
            if 0.0 <= result[2][i] < 0.2:
                prob_counter[0] += 1
            elif 0.2 <= result[2][i] < 0.4:
                prob_counter[1] += 1
            elif 0.4 <= result[2][i] < 0.6:
                prob_counter[2] += 1
            elif 0.6 <= result[2][i] < 0.8:
                prob_counter[3] += 1
            elif 0.8 <= result[2][i] <= 1.0:
                prob_counter[4] += 1
            sum_probabilities += result[2][i]
        # print('Sum of probabilities: ' + str(sum_probabilities))
        print('Mean of probabilities: ' + str(sum_probabilities / len(result[0])))
        print('Counters: ')
        print('[0-20): ' + str(prob_counter[0]) + '; (' + str(round(prob_counter[0]/len(result[0]), 2)) + '%)')
        print('[20-40): ' + str(prob_counter[1]) + '; (' + str(round(prob_counter[1]/len(result[0]), 2)) + '%)')
        print('[40-60): ' + str(prob_counter[2]) + '; (' + str(round(prob_counter[2]/len(result[0]), 2)) + '%)')
        print('[60-80): ' + str(prob_counter[3]) + '; (' + str(round(prob_counter[3]/len(result[0]), 2)) + '%)')
        print('[80-100]: ' + str(prob_counter[4]) + '; (' + str(round(prob_counter[4]/len(result[0]), 2)) + '%)')
        print('All: ' + str(len(result[0])))

def read_dataset_train_from_csv():
    dataset_train = pd.read_csv(dataset_train_file)
    # dataset_train = shuffle(dataset_train)
    X_train = dataset_train.drop(['class', 'song_filename'], axis=1)
    y_train = dataset_train['class']
    return X_train, y_train

def read_dataset_test_from_csv():
    dataset_test = pd.read_csv(dataset_test_file)
    X_test = dataset_test.drop(['song_filename'], axis=1)
    filenames_test = dataset_test['song_filename']
    return X_test, filenames_test

def learn_by_DecisionTreeClassifier(X_train_scl, y_train, X_test_scl):
    print('DecisionTree:')
    # kf = KFold(n_splits=5, shuffle=True)
    dtc = DecisionTreeClassifier()
    # print('[MESSAGE]: Please, wait! Start cross-validation...')
    # quality = cross_val_score(dtc, X_train_scl, y_train, cv=kf, scoring='accuracy')
    # print('[MESSAGE]: Cross-validation finished successfully!')
    # print('Quality: ' + str(quality))
    print('[MESSAGE]: Please, wait! Start learning by DecisionTree...')
    dtc.fit(X_train_scl, y_train)
    print('[MESSAGE]: Learning finished successfully!')
    y_test = dtc.predict(X_test_scl)
    print('Predict: ' + str(y_test) + '\n')
    probabilities = get_probabilities_for_predict(X_test_scl, y_test, dtc)

    return y_test, probabilities

def learn_by_kNeighborsClassifier(X_train_scl, y_train, X_test_scl):
    print('kNeighbors:')
    # kf = KFold(n_splits=5, shuffle=True)
    neigh = KNeighborsClassifier(n_neighbors=2)
    # print('[MESSAGE]: Please, wait! Start cross-validation...')
    # quality = cross_val_score(neigh, X_train_scl, y_train, cv=kf, scoring='accuracy')
    # print('[MESSAGE]: Cross-validation finished successfully!')
    # print('Quality: ' + str(quality))
    print('[MESSAGE]: Please, wait! Start learning by kNeighbors...')
    neigh.fit(X_train_scl, y_train)
    print('[MESSAGE]: Learning finished successfully!')
    y_test = neigh.predict(X_test_scl)
    print('Predict: ' + str(y_test) + '\n')
    probabilities = get_probabilities_for_predict(X_test_scl, y_test, neigh)

    return y_test, probabilities

def learn_by_SVC(X_train_scl, y_train, X_test_scl):
    print('SVC:')
    # kf = KFold(n_splits=5, shuffle=True)
    svc = SVC(C=10.0, probability=True)
    # print('[MESSAGE]: Please, wait! Start cross-validation...')
    # quality = cross_val_score(svc, X_train_scl, y_train, cv=kf, scoring='accuracy')
    # print('[MESSAGE]: Cross-validation finished successfully!')
    # print('Quality: ' + str(quality))
    print('[MESSAGE]: Please, wait! Start learning by SVC...')
    svc.fit(X_train_scl, y_train)
    print('[MESSAGE]: Learning finished successfully!')
    y_test = svc.predict(X_test_scl)
    print('Predict: ' + str(y_test) + '\n')
    probabilities = get_probabilities_for_predict(X_test_scl, y_test, svc)

    return y_test, probabilities

def learn_by_RandomForestClassifier(X_train_scl, y_train, X_test_scl):
    print('RandomForest:')
    # kf = KFold(n_splits=5, shuffle=True)
    rfc = RandomForestClassifier(n_estimators=150)
    # print('[MESSAGE]: Please, wait! Start cross-validation...')
    # quality = cross_val_score(rfc, X_train_scl, y_train, cv=kf, scoring='accuracy')
    # print('[MESSAGE]: Cross-validation finished successfully!')
    # print('Quality: ' + str(quality))
    print('[MESSAGE]: Please, wait! Start learning by RandomForest...')
    rfc.fit(X_train_scl, y_train)
    print('[MESSAGE]: Learning finished successfully!')
    y_test = rfc.predict(X_test_scl)
    print('Predict: ' + str(y_test) + '\n')
    probabilities = get_probabilities_for_predict(X_test_scl, y_test, rfc)

    return y_test, probabilities

def learn_by_GradientBoostingClassifier(X_train_scl, y_train, X_test_scl):
    print('GradientBoosting:')
    # kf = KFold(n_splits=5, shuffle=True)
    gbc = GradientBoostingClassifier(n_estimators=150, learning_rate=0.2)
    # print('[MESSAGE]: Please, wait! Start cross-validation...')
    # quality = cross_val_score(gbc, X_train_scl, y_train, cv=kf, scoring='accuracy')
    # print('[MESSAGE]: Cross-validation finished successfully!')
    # print('Quality: ' + str(quality))
    print('[MESSAGE]: Please, wait! Start learning by GradientBoosting...')
    gbc.fit(X_train_scl, y_train)
    print('[MESSAGE]: Learning finished successfully!')
    y_test = gbc.predict(X_test_scl)
    print('Predict: ' + str(y_test) + '\n')
    probabilities = get_probabilities_for_predict(X_test_scl, y_test, gbc)

    return y_test, probabilities

def get_probabilities_for_predict(X_test_scl, y_test, clf):
    predict_proba = clf.predict_proba(X_test_scl)
    probabilities = []
    if type(y_test[0]) == type(''):
        dict = {'1_':1, '2_':2, '3_':3, '4_':4, '5_':5}
        for i in range(len(y_test)):
            y_test[i] = dict[y_test[i]]

    for i, predict in enumerate(y_test):
        probabilities.append(predict_proba[i][predict - 1])

    return probabilities


path_to_root = str(os.getcwd()) + '\\'
dataset_train_file = 'dataset_train.csv'
dataset_test_file = 'dataset_test.csv'
# dataset_train_file = 'dataset_norm_train.csv'
# dataset_test_file = 'dataset_norm_test.csv'
# dataset_train_file = 'dataset1_train.csv'
# dataset_test_file = 'dataset1_test.csv'
# dataset_train_file = 'dataset1_norm_train.csv'
# dataset_test_file = 'dataset1_norm_test.csv'

result_file_dtc = 'result_dtc.csv'
result_file_neigh = 'result_neigh.csv'
result_file_svc = 'result_svc.csv'
result_file_rfc = 'result_rfc.csv'
result_file_gbc = 'result_gbc.csv'
# result_file_dtc = 'result_norm_dtc.csv'
# result_file_neigh = 'result_norm_neigh.csv'
# result_file_svc = 'result_norm_svc.csv'
# result_file_rfc = 'result_norm_rfc.csv'
# result_file_gbc = 'result_norm_gbc.csv'
# result_file_dtc = 'result1_dtc.csv'
# result_file_neigh = 'result1_neigh.csv'
# result_file_svc = 'result1_svc.csv'
# result_file_rfc = 'result1_rfc.csv'
# result_file_gbc = 'result1_gbc.csv'
# result_file_dtc = 'result1_norm_dtc.csv'
# result_file_neigh = 'result1_norm_neigh.csv'
# result_file_svc = 'result1_norm_svc.csv'
# result_file_rfc = 'result1_norm_rfc.csv'
# result_file_gbc = 'result1_norm_gbc.csv'

classes = '1 2 3 4 5'.split()

# minimum_duration = 2.9
dataset_test_folder = 'test_data'

create_train_csv()
train_to_csv()

X_train, y_train = read_dataset_train_from_csv()
scale = StandardScaler()
X_train_scl = scale.fit_transform(X_train)

create_test_csv()
test_to_csv()

X_test, filenames_test = read_dataset_test_from_csv()
print('\nX_test:')
print(X_test)
print('------------------')
X_test_scl = scale.transform(X_test)

y_test_dtc, probabilities_dtc = learn_by_DecisionTreeClassifier(X_train_scl, y_train, X_test_scl)
y_test_neigh, probabilities_neigh = learn_by_kNeighborsClassifier(X_train_scl, y_train, X_test_scl)
y_test_svc, probabilities_svc = learn_by_SVC(X_train_scl, y_train, X_test_scl)
y_test_rfc, probabilities_rfc = learn_by_RandomForestClassifier(X_train_scl, y_train, X_test_scl)
y_test_gbc, probabilities_gbc = learn_by_GradientBoostingClassifier(X_train_scl, y_train, X_test_scl)

result_dtc = [filenames_test, y_test_dtc, probabilities_dtc]
result_neigh = [filenames_test, y_test_neigh, probabilities_neigh]
result_svc = [filenames_test, y_test_svc, probabilities_svc]
result_rfc = [filenames_test, y_test_rfc, probabilities_rfc]
result_gbc = [filenames_test, y_test_gbc, probabilities_gbc]

output_results_to_console()

create_results_csvs()
output_results_to_csvs()