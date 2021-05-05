import numpy as np
import math
import os
import pandas as pd
from dtw import *
from hmmlearn import hmm
from sklearn.model_selection import KFold
import warnings
import random
warnings.filterwarnings("ignore")


# tanh normalization; values between 0 and 1
def normalization(data):
    mean = data.mean()
    std = data.std() + np.finfo(np.float32).eps
    data = 0.5 * (np.tanh(0.01 * ((data - mean) / std)) + 1)
    return data


def z_normalization(df):
    column_maxes = df.max()
    df_max = column_maxes.max()
    return df / df_max


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

    training_list = list()  # dictionary of all training dataframes [1..40]
    testing_dict = {'skilled': [], 'genuine': []}  # dictionary of all testing dataframes [41..56]
    training_fs_list = list()  # list of training dataframes (feature selection) [alltrain - validation]
    validation_fs_dict = {'true': [], 'false': []}  # dictionary of validation dataframes (feature selection) [one true for each session, 18 false random]
    random.seed(42)  # pseudorandomic
    path = './xLongSignDB/'
    path_user = path + str(user) + '/'
    files = os.listdir(path_user)
    i = 0

    for file in files:
        df = pd.read_csv(path_user + file, header=0, sep=' ', names=['X', 'Y', 'TIMESTAMP', 'PENSUP', 'AZIMUTH', 'ALTITUDE', 'Z'])
        df = initialize_dataset(df)

        if 'ss' in file:
            testing_dict['skilled'].append(df)

        elif 'sg' in file:
            i += 1

            if i > 41:
                testing_dict['genuine'].append(df)
            else:
                training_list.append(df)
                if i % 4 == 0:
                    validation_fs_dict['true'].append(df)
                else:
                    training_fs_list.append(df)

        else:
            print(file)

    for i in range(0, 20):
        numbers = [x for x in range(1, 30)]
        numbers.remove(user)
        y = random.choice(numbers)
        path_user = path + str(y) + '/'
        files = os.listdir(path_user)
        z = random.randint(10, 40)
        df = pd.read_csv(path_user + files[z], header=0, sep=' ', names=['X', 'Y', 'TIMESTAMP', 'PENSUP', 'AZIMUTH', 'ALTITUDE', 'Z'])
        df = initialize_dataset(df)
        validation_fs_dict['false'].append(df)

    return training_list, testing_dict, training_fs_list, validation_fs_dict


def initialize_dataset(df):
    del df['TIMESTAMP']
    del df['AZIMUTH']
    del df['ALTITUDE']
    df = compute_features(df)
    df = normalization(df)
    return df


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

    del df['PENSUP']
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

    return df


def feature_selection(training_set, validation_set):

    header = list(training_set[0].columns.values)
    k = 0  # counter number of feature to select
    subset = set()  # empty set ("null set") so that the k = 0 (where k is the size of the subset)
    total_features = set(header)

    while k != 9:

        best_score = 1
        best_feature = ""
        feature_set = subset.copy()

        for f in (total_features - subset):
            feature_set.add(f)
            score = evaluate_score(training_set, list(feature_set), validation_set, header)
            feature_set.remove(f)
            if score < best_score:
                best_score = score
                best_feature = f

        subset.add(best_feature)
        worst_score = 1
        worst_feature = ""
        feature_set = subset.copy()
        print("best "+str(best_feature))

        if len(subset) > 1:

            for f in subset:
                feature_set.remove(f)
                score = evaluate_score(training_set, list(feature_set), validation_set, header)
                feature_set.add(f)
                if score < worst_score:
                    worst_score = score
                    worst_feature = f

            if worst_feature == best_feature:
                k += 1
            else:
                subset.remove(worst_feature)
                print("removed"+str(worst_feature))
        else:
            k += 1

    return subset


def evaluate_score(training_set, features, validation_set, header):

    print(features)

    y_train = [len(x) for x in training_set]

    train_df = np.concatenate(training_set)
    train_df = pd.DataFrame(data=train_df, columns=header)
    train_df = train_df[features]

    # test_df = np.concatenate(validation_set)
    # test_df = pd.DataFrame(data=test_df, columns=header)
    # test_df = test_df[features]

    score_train = 0
    score_test = 0
    average_score = 0
    result = 0
    count_training = 0
    count_validation = 0
    min_score = np.inf
    false_acceptance = 0
    false_rejection = 0

    try:

        model = hmm.GMMHMM(n_components=32, n_mix=2, random_state=42)

        model.fit(train_df, y_train)

        for signature in training_set:
            a = pd.DataFrame(data=signature, columns=header)
            a = a[features]
            score_train = model.score(a)
            if score_train < min_score:
                min_score = score_train
            average_score += score_train
            count_training += 1

        average_score /= 31
        distance = np.abs(min_score - score_train)
        threshold = np.exp(distance * (-1) / len(features))

        for signature in validation_set["true"]:
            a = pd.DataFrame(data=signature, columns=header)
            a = a[features]
            distance = np.abs(model.score(a) - score_train)
            score_test = np.exp(distance*(-1)/len(features))
            count_validation += 1
            if score_test < threshold:
                false_rejection += 1

        for signature in validation_set["false"]:
            a = pd.DataFrame(data=signature, columns=header)
            a = a[features]
            distance = np.abs(model.score(a) - score_train)
            score_test = np.exp(distance*(-1)/len(features))
            count_validation += 1
            if score_test >= threshold:
                false_acceptance += 1

        # probabilità da chiedere
        false_acceptance_rate = false_acceptance / len(validation_set["false"])
        false_rejection_rate = false_rejection / len(validation_set["true"])
        probability_false = len(validation_set["false"])/(len(validation_set["false"])+len(validation_set["true"]))
        probability_true = 1 - probability_false
        equal_error_rate = (false_rejection_rate * probability_true) + (false_acceptance_rate * probability_false)
        print(f"false acceptance: {false_acceptance}; false rejection: {false_rejection}; "
              f"far: {false_rejection_rate}; frr: {false_rejection_rate};");

    except:

        equal_error_rate = 1
        print("Fit Training Error")


    print(f"equal error rate: {equal_error_rate}")
    return equal_error_rate


training_list, testing_dict, training_fs_list, validation_fs_dict = load_dataset(1)
feature_selection(training_fs_list, validation_fs_dict)