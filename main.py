import tkinter.tix

import numpy as np
import math
import csv
import os
import pandas as pd
from dtw import *
from sklearn.model_selection import train_test_split
import warnings
import random
from pomegranate import *
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")
np.random.seed(42)  # pseudorandomic

# tanh normalization; values between 0 and 1
def normalization(data):
    mean = data.mean()
    std = data.std()
    data = 0.5 * (np.tanh(0.01 * ((data - mean) / std)) + 1)
    return data


def z_normalization(df):
    column_maxes = df.max()
    df_max = column_maxes.max()
    return df / df_max

def norm(A):
    norm = (A-np.min(A))/(np.max(A) - np.min(A))
    return norm

# time derivative approsimation
def second_order_regression(data_list):
    pad_list = data_list.copy()
    pad_list.append(data_list[len(data_list) - 1])
    pad_list.append(data_list[len(data_list) - 1])
    pad_list.insert(0, data_list[0])
    pad_list.insert(0, data_list[0])
    derivata = [None] * (len(data_list))

    for n in range(2, len(data_list) + 2):
        derivata[n - 2] = ((pad_list[n + 1] - pad_list[n - 1] + 2 * (pad_list[n + 2] - pad_list[n - 2])) / 10)

    return derivata


def load_dataset(user):

    training_list = list()  # list of all training dataframes [1..40]
    testing_dict = {'skilled': [], 'genuine': []}  # dictionary of all testing dataframes [41..56]

    random.seed(42)  # pseudorandomic
    path = './xLongSignDB/'
    path_user = path + str(user) + '/'
    files = os.listdir(path_user)
    i = 0

    for file in files:
        df = pd.read_csv(path_user + file, header=0, sep=' ',
                         names=['X', 'Y', 'TIMESTAMP', 'PENSUP', 'AZIMUTH', 'ALTITUDE', 'Z'])
        df = initialize_dataset(df)

        if 'ss' in file:
            testing_dict['skilled'].append(df)

        elif 'sg' in file:
            i += 1

            if i > 41:
                testing_dict['genuine'].append(df)
            else:
                training_list.append(df)
        else:
            print(file)

    x_train_fs, x_validation_fs = train_test_split(training_list, test_size = 0.2, random_state = 42)

    return training_list, testing_dict, x_train_fs, x_validation_fs


def initialize_dataset(df):
    del df['TIMESTAMP']
    del df['AZIMUTH']
    del df['ALTITUDE']
    df = compute_features(df)
    new_df = pd.DataFrame(data=None)
    for column in df.columns.values:
        new_df[column] = normalization(df[column].values)
    #df = normalization(df)
    return new_df


def compute_features(df):
    e = np.finfo(np.float32).eps
    leng = len(df)

    x_list = df['X'].tolist()
    y_list = df['Y'].tolist()
    z_list = df['Z'].tolist()  # pression
    up_list = df['PENSUP'].tolist()

    dx_list = second_order_regression(x_list)
    dy_list = second_order_regression(y_list)

    theta_list = [None] * leng
    v_list = [None] * leng  # velocity

    for i in range(0, leng):
        theta_list[i] = np.arctan(dy_list[i] / (dx_list[i] + e))
        v_list[i] = np.sqrt(dy_list[i] ** 2 + dx_list[i] ** 2)

    dtheta_list = second_order_regression(theta_list)
    p_list = [None] * leng  # ro
    dv_list = second_order_regression(v_list)
    a_list = [None] * leng  # acceleration

    for i in range(0, leng):
        dtheta = np.abs(dtheta_list[i] + e)
        v = np.abs(v_list[i] + e)

        p_list[i] = math.log(v / dtheta)
        a_list[i] = np.sqrt((dtheta_list[i] * v_list[i]) ** 2 + dv_list[i] ** 2)

    dz_list = second_order_regression(z_list)
    da_list = second_order_regression(a_list)
    dp_list = second_order_regression(p_list)

    ddx_list = second_order_regression(dx_list)
    ddy_list = second_order_regression(dy_list)

    ratios_list = [None] * leng  # np.array([])  Ratio of the minimum over the maximum speed over a 5-samples window

    for n in range(0, leng - 4):
        vmin = np.amin(v_list[n:n + 4])
        vmax = np.amax(v_list[n:n + 4]) + e
        ratios_list[n] = vmin / vmax

    alphas_list = [None] * leng  # Angle of consecutive samples

    for n in range(0, leng - 1):
        alphas_list[n] = np.arctan((y_list[n + 1] - y_list[n]) / (x_list[n + 1] - x_list[n] + e))

    alphas_list[leng - 1] = alphas_list[leng - 2]

    dalpha_list = second_order_regression(alphas_list)
    sinalpha_list = np.array(np.sin(alphas_list))
    cosalpha_list = np.array(np.cos(alphas_list))

    strokeratio5_list = [None] * leng  # Stroke length to width ratio over a 5-samples window

    for n in range(0, leng - 4):
        stroke_len = np.sum(up_list[n:n + 4])
        width = np.max(x_list[n:n + 4]) - np.min(x_list[n:n + 4]) + e
        strokeratio5_list[n] = stroke_len / width

    strokeratio7_list = [None] * leng  # Stroke length to width ratio over a 7-samples window

    for n in range(0, leng - 6):
        stroke_len = np.sum(up_list[n:n + 6])
        width = np.max(x_list[n:n + 6]) - np.min(x_list[n:n + 6]) + e
        strokeratio7_list[n] = stroke_len / width

    df['theta'] = theta_list
    df['v'] = v_list
    df['ro'] = p_list
    df['ac'] = a_list
    df['dX'] = dx_list
    df['dY'] = dy_list
    df['dZ'] = dz_list
    df['dTheta'] = dtheta_list
    df['dv'] = dv_list
    df['dRo'] = dp_list
    df['dAc'] = da_list
    df['ddX'] = ddx_list
    df['ddY'] = ddy_list
    df['vRatio'] = ratios_list
    df['alpha'] = alphas_list
    df['dAlpha'] = dalpha_list
    df['SIN'] = sinalpha_list
    df['COS'] = cosalpha_list
    df['5_WIN'] = strokeratio5_list
    df['7_WIN'] = strokeratio7_list

    df['vRatio'].fillna(value=df['vRatio'].mean(), inplace=True)
    df['alpha'].fillna(value=df['alpha'].mean(), inplace=True)
    df['dAlpha'].fillna(value=df['dAlpha'].mean(), inplace=True)
    df['SIN'].fillna(value=df['SIN'].mean(), inplace=True)
    df['COS'].fillna(value=df['COS'].mean(), inplace=True)
    df['5_WIN'].fillna(value=df['5_WIN'].mean(), inplace=True)
    df['7_WIN'].fillna(value=df['7_WIN'].mean(), inplace=True)

    # del(df['PENSUP'])

    return df

def feature_selection(training_set, validation_set):
    k = 0  # counter number of feature to select
    subset = set()  # empty set ("null set") so that the k = 0 (where k is the size of the subset)
    header = list(training_set[0].columns.values)
    total_features = set(header)
    last_score = [1] * 10
    while k != 9:

        best_score = 0
        best_feature = ""
        feature_set = subset.copy()

        for f in (total_features - subset):
            feature_set.add(f)
            score = evaluate_score(training_set,validation_set, list(feature_set), )
            feature_set.remove(f)
            if score > best_score:
                best_score = score
                best_feature = f

        subset.add(best_feature)
        k += 1
        last_score[k] = best_score

        print("best " + str(best_feature))

        still_remove = True
        while still_remove:

            if len(subset) > 1:

                feature_set = subset.copy()

                worst_feature = ""
                removed_score = 0
                for f in subset:
                    feature_set.remove(f)
                    score = evaluate_score(training_set, validation_set,list(feature_set))
                    feature_set.add(f)
                    if score > removed_score:
                        removed_score = score
                        worst_feature = f

                if removed_score > last_score[k-1]:  # if you get better than before adding, keep removing features
                    last_score[k-1] = removed_score
                    subset.remove(worst_feature)
                    k -= 1
                    print("removed" + str(worst_feature))
                    still_remove = True
                else:                           # otherwhise removed score is worse than before, go adding features
                    still_remove = False
            else:
                k = 1
                still_remove = False

    return subset, best_score


def evaluate_score(training_set,validation_set,features):

    print(features)

    train_set = training_set.copy()

    i = 0
    avg_accuracy = 0
    for signature in train_set:
        train_set[i] = signature[features]
        i += 1

    label_train = pd.DataFrame(data=[1] * len(train_set), columns=['Y'])
    label_validation = pd.DataFrame(data=[1] * len(validation_set), columns=['Y'])

    try:
        score_train = [None] * (len(train_set))
        score_validation = [None] * (len(validation_set))
        count_training = 0
        count_validation = 0

        model = HiddenMarkovModel.from_samples(NormalDistribution,
                                               n_components=2,
                                               X=np.array(train_set),
                                               random_state=42)

        for signature in train_set:
            score_train[count_training] = model.score(np.array(signature),np.array(label_train))
            count_training += 1

        for signature in validation_set:
            a = signature[features]
            score_validation[count_validation] = model.score(np.array(a),np.array(label_validation))
            count_validation += 1

        i = 0
        for score in score_train:
            i += 1
            #print(f"accuracy on training for signature n. {i}: {score}")

        i = 0
        for score in score_validation:
            i += 1
            #print(f"accuracy on testing for signature n. {i}: {score}")

        avg_accuracy_train = np.mean(score_train)
        avg_accuracy = np.mean(score_validation)
        print(f"average accuracy on training: {avg_accuracy_train}")
        print(f"average accuracy on testing:  {avg_accuracy}")

    except:
        print("Fit error")

    return avg_accuracy


def test_evaluation(training_set, features, validation_set):

    i = 0

    for signature in training_set:
        training_set[i] = signature[features]
        i += 1

    try:

        model = HiddenMarkovModel.from_samples(NormalDistribution,
                                               n_components=2,
                                               X=np.array(training_set),
                                               random_state=42,
                                               n_init=4,
                                               inertia=1.0)

        score_train = [None] * (len(training_set))
        score_test_gen = [None] * (len(validation_set["genuine"]))
        score_test_skilled = [None] * (len(validation_set["skilled"]))

        count_training = 0
        for signature in training_set:
            score_train[count_training] = model.log_probability(signature)
            count_training += 1

        average_score = np.average(score_train)
        count_training = 0

        for signature in training_set:
            a = signature[features]
            distance = np.abs(model.log_probability(a) - average_score)
            score_train[count_training] = np.exp(distance/len(features))
            count_training += 1

        count_validation = 0

        for signature in validation_set["genuine"]:
            a = signature[features]
            distance = np.abs(model.log_probability(a) - average_score)
            score_test_gen[count_validation] = np.exp((distance * -1) / len(features))
            count_validation += 1

        count_validation = 0

        for signature in validation_set["skilled"]:
            a = signature[features]
            distance = np.abs(model.log_probability(a) - average_score)
            score_test_skilled[count_validation] = np.exp((distance * -1) / len(features))
            count_validation += 1

        i = 0
        for score in score_test_gen:
            i += 1
            print(f" prob signature testing genuine {i}: {score}")

        i = 0
        for score in score_test_skilled:
            i += 1
            print(f" prob signature testing skilled {i}: {score}")

        labels = [1] * 5 + [0] * 10
        probs = np.concatenate([score_test_gen, score_test_skilled])
        fpr, tpr, thresh = roc_curve(labels, probs)
        equal_error_rate = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        threshold = interp1d(fpr, thresh)(equal_error_rate)
        print(f"threshold: {threshold} eer: {equal_error_rate}")

    except:
        equal_error_rate = 0.5
        print(f"ERROR Nan values eer: {equal_error_rate}")

    return equal_error_rate


def testing():

    score_exp1 = [None] * 29
    score_exp2 = [None] * 29
    score_exp3 = [None] * 29
    score_exp4 = [None] * 29
    score_exp5 = [None] * 29
    score_exp6 = [None] * 29
    score_exp_g = [None] * 29
    score_exp_h = [None] * 29
    score_exp_i = [None] * 29
    score_exp_j = [None] * 29
    score_exp_k = [None] * 29
    score_exp_l = [None] * 29
    score_exp_m = [None] * 29
    score_exp_n = [None] * 29
    score_exp_o = [None] * 29

    with open('features_sffs.csv', mode='r') as feature_file:
        feature_reader = csv.reader(feature_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        i = 0
        for (fs) in feature_reader:
            if i > 0:

                print(f"user n#{i}")
                print(fs)
                fs.pop()

                training_list, testing_dict, training_fs_list, validation_fs_dict = load_dataset(i)

                training = training_list[0:4]
                score_exp1[i - 1] = test_evaluation(training, fs, testing_dict)

                training = training_list[4:8]
                score_exp2[i - 1] = test_evaluation(training, fs, testing_dict)

                training = training_list[8:12]
                score_exp3[i - 1] = test_evaluation(training, fs, testing_dict)

                training = training_list[12:16]
                score_exp4[i - 1] = test_evaluation(training, fs, testing_dict)

                training = training_list[21:25]
                score_exp5[i - 1] = test_evaluation(training, fs, testing_dict)

                training = training_list[36:40]
                score_exp6[i - 1] = test_evaluation(training, fs, testing_dict)

                score_exp_g[i - 1]  = score_exp1[i - 1]

                training = training_list[0:16]
                score_exp_h[i - 1] = test_evaluation(training, fs, testing_dict)

                training = training_list[0:31]
                score_exp_i[i - 1] = test_evaluation(training, fs, testing_dict)

                training = training_list[1:16] + training_list[21:25]
                score_exp_j[i - 1] = test_evaluation(training, fs, testing_dict)

                training = training_list[4:16] + training_list[21:25]
                score_exp_k[i - 1] = test_evaluation(training, fs, testing_dict)

                training = training_list[8:16] + training_list[21:25]
                score_exp_l[i - 1] = test_evaluation(training, fs, testing_dict)

                training = training_list[12:16] + training_list[21:25]
                score_exp_m[i - 1] = test_evaluation(training, fs, testing_dict)

                score_exp_n[i - 1] = score_exp5[i - 1]

                training = training_list[16:31]
                score_exp_o[i - 1] = test_evaluation(training, fs, testing_dict)

            i += 1

        eer1 = np.average(score_exp1)
        eer2 = np.average(score_exp2)
        eer3 = np.average(score_exp3)
        eer4 = np.average(score_exp4)
        eer5 = np.average(score_exp5)
        eer6 = np.average(score_exp6)
        eer_g = np.average(score_exp_g)
        eer_h = np.average(score_exp_h)
        eer_i = np.average(score_exp_i)
        eer_j = np.average(score_exp_j)
        eer_k = np.average(score_exp_k)
        eer_l = np.average(score_exp_l)
        eer_m = np.average(score_exp_m)
        eer_n = np.average(score_exp_n)
        eer_o = np.average(score_exp_o)

        print("average equal error rate for experiment:")
        print(eer1, eer2, eer3, eer4, eer5, eer6, eer_g, eer_h, eer_i, eer_j, eer_k, eer_l, eer_m, eer_n, eer_o)



def start_fs():
    for i in range(1, 30):

        training_list, testing_dict, training_fs_list, validation_fs_dict = load_dataset(i)
        subset, eer = feature_selection(training_fs_list, validation_fs_dict)
        # subset, eer = feature_selection_accuracy(training_fs_list, validation_fs_dict["true"])
        with open('features_accuracy.csv', mode='a') as feature_file:
            feature_writer = csv.writer(feature_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            subset = list(subset)
            subset.append(eer)
            feature_writer.writerow(subset)

start_fs()

'''

testing()

'''