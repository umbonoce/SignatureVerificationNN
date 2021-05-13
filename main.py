import numpy as np
import math
import os
import pandas as pd
from dtw import *
from hmmlearn import hmm
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import warnings
import random
from pomegranate import *
warnings.filterwarnings("ignore")
import csv


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
    validation_fs_dict = {'true': [],
                          'false': []}  # dictionary of validation dataframes (feature selection) [one true for each session, 18 false random]
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
        df = pd.read_csv(path_user + files[z], header=0, sep=' ',
                         names=['X', 'Y', 'TIMESTAMP', 'PENSUP', 'AZIMUTH', 'ALTITUDE', 'Z'])
        df = initialize_dataset(df)
        validation_fs_dict['false'].append(df)

    return training_list, testing_dict, training_fs_list, validation_fs_dict


def initialize_dataset(df):
    del df['TIMESTAMP']
    del df['AZIMUTH']
    del df['ALTITUDE']
    df = compute_features(df)
    # df = normalization(df)
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
        print("best " + str(best_feature))

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
                print("removed" + str(worst_feature))
        else:
            k += 1

    return subset, best_score


def evaluate_score(train_set, features, validation_set, header):

    # training_set = np.concatenate(training_set)
    # train_set = pd.DataFrame(data=training_set, columns=header)
    # train_set = train_set[features]
    print(header)
    print(features)

    i = 0
    training_set = train_set.copy()
    for signature in training_set:
        training_set[i] = pd.DataFrame(data=signature, columns=features)
        print(training_set[i])
        i += 1

    # train_df = np.concatenate(training_set)
    # train_df = pd.DataFrame(data=train_df, columns=header)
    # train_df = train_df[features]

    score_train = [None] * (len(training_set))
    count_training = 0
    count_validation = 0
    false_acceptance = 0
    false_rejection = 0

    model = HiddenMarkovModel.from_samples(NormalDistribution, n_components=2, X = np.array(training_set))

    for signature in training_set:
        a = pd.DataFrame(data=signature, columns=header)
        a = a[features]
        score_train[count_training] = model.log_probability(np.array(a))
        count_training += 1

    average_score = np.mean(score_train)
    min_score = np.min(score_train)
    distance = np.abs(min_score - average_score)
    threshold = np.exp(distance * (-1) / len(features))
    print(threshold)

    if threshold is not None:

        for signature in validation_set["true"]:
            a = pd.DataFrame(data=signature, columns=header)
            a = a[features]
            distance = np.abs(model.log_probability(np.array(a)) - average_score)
            score_test = np.exp(distance * (-1) / len(features))
            count_validation += 1

            if score_test < threshold:
                false_rejection += 1

        for signature in validation_set["false"]:
            a = pd.DataFrame(data=signature, columns=header)
            a = a[features]
            distance = np.abs(model.log_probability(np.array(a)) - average_score)
            score_test = np.exp(distance * (-1) / len(features))
            count_validation += 1

            if score_test >= threshold:
                false_acceptance += 1

        false_acceptance_rate = false_acceptance / len(validation_set["false"])
        false_rejection_rate = false_rejection / len(validation_set["true"])
        probability_false = len(validation_set["false"]) / (len(validation_set["false"]) + len(validation_set["true"]))
        probability_true = 1 - probability_false
        equal_error_rate = (false_rejection_rate * probability_true) + (false_acceptance_rate * probability_false)
        print(f"false acceptance: {false_acceptance}; false rejection: {false_rejection}; "
              f"far: {false_rejection_rate}; frr: {false_rejection_rate};");

    else:

        equal_error_rate = 1
        print("Fit Training Error")

    print(f"equal error rate: {equal_error_rate}")
    return equal_error_rate


def test_evaluation(training_set, features, validation_set):

    i = 0

    for signature in training_set:
        training_set[i] = normalization(signature[features])
        i += 1

    score_train = [None] * (len(training_set))
    score_test_gen = [None] * (len(validation_set["genuine"]))
    score_test_skilled = [None] * (len(validation_set["skilled"]))
    count_training = 0
    count_validation = 0
    false_acceptance = 0
    false_rejection = 0

    model = HiddenMarkovModel.from_samples(NormalDistribution, n_components=2, X=training_set)

    for signature in training_set:
        score_train[count_training] = model.log_probability(signature)
        # print(signature)
        # print(f" prob signature {count_training}: {score_train[count_training]}")
        count_training += 1

    average_score = np.average(score_train)
    mu = np.mean(score_train)
    std_dev = np.array(score_train).std()

    # min_score = np.min(score_train) * 0.95
    # min_score = min_score - (3 * std_dev / len(score_train))

    limit_min = mu - 3 * std_dev # 3 sigma rule
    min_score = np.min(score_train)

    print(f"3 sigma limit min: {limit_min}; min score{min_score}")
    # min_score = (limit_min + min_score) / 2

    #if len(score_train) == 4:
    #    min_score = (limit_min * 3 + min_score) / 4
    #else:
    #    min_score = (limit_min + min_score * 3) / 4

    distance = np.abs(min_score - average_score)
    threshold = np.exp(distance * (-1) / len(features))

    print(f"prob threshold: {threshold}")

    for signature in validation_set["genuine"]:
        a = signature[features]
        a = normalization(a)
        score_test_gen[count_validation] = model.log_probability(a)
        distance = np.abs(score_test_gen[count_validation] - average_score)
        score_test = np.exp(distance * (-1) / len(features))
        print(f" prob signature genuine {count_validation+1}: {score_test}")
        count_validation += 1
        # print(f"distance score on genuine: {score_test}, {threshold}")

        if score_test <= threshold:
            false_rejection += 1

    count_validation = 0

    for signature in validation_set["skilled"]:
        a = signature[features]
        a = normalization(a)
        score_test_skilled[count_validation] = model.log_probability(a)
        distance = np.abs(score_test_skilled[count_validation] - average_score)
        score_test = np.exp(distance * (-1) / len(features))
        print(f" prob signature skilled {count_validation+1}: {score_test}")
        count_validation += 1
        # print(f"distance score on skilled: {score_test}, {threshold}")
        if score_test > threshold:
            false_acceptance += 1

    false_acceptance_rate = false_acceptance / len(validation_set["skilled"])
    false_rejection_rate = false_rejection / len(validation_set["genuine"])
    probability_false = len(validation_set["skilled"]) / (
                len(validation_set["skilled"]) + len(validation_set["genuine"]))
    probability_true = 1 - probability_false
    equal_error_rate = (false_rejection_rate * probability_true) + (
                false_acceptance_rate * probability_false)

    print(f"false acceptance {false_acceptance}, false rejection {false_rejection},eer{equal_error_rate}")

    '''
    length_train = len(score_train)
    length_val = len(score_test_gen) + length_train
    length_test = len(score_test_skilled) + length_val
    plt.figure(figsize=(7, 5))
    plt.scatter(np.arange(length_train), score_train, c='b', label='trainset')
    plt.scatter(np.arange(length_train, length_val), score_test_gen, c='g', label='testset - original')
    plt.scatter(np.arange(length_val, length_test), score_test_skilled, c='r', label='testset - skilled')
    plt.title(f'User: {user} | HMM states:{components}  | GMM components: 2')
    plt.show()'''

    return [equal_error_rate, false_acceptance_rate, false_rejection_rate]


score_exp1, far1, frr1 = [None] * 29, [None] * 29, [None] * 29
score_exp2, far2, frr2 = [None] * 29, [None] * 29, [None] * 29
score_exp3, far3, frr3 = [None] * 29, [None] * 29, [None] * 29
score_exp4, far4, frr4 = [None] * 29, [None] * 29, [None] * 29
score_exp5, far5, frr5 = [None] * 29, [None] * 29, [None] * 29
score_exp6, far6, frr6 = [None] * 29, [None] * 29, [None] * 29
score_exp_g, far_g, frr_g = [None] * 29, [None] * 29, [None] * 29
score_exp_h, far_h, frr_h = [None] * 29, [None] * 29, [None] * 29
score_exp_i, far_i, frr_i = [None] * 29, [None] * 29, [None] * 29
score_exp_j, far_j, frr_j = [None] * 29, [None] * 29, [None] * 29
score_exp_k, far_k, frr_k = [None] * 29, [None] * 29, [None] * 29
score_exp_l, far_l, frr_l = [None] * 29, [None] * 29, [None] * 29
score_exp_m, far_m, frr_m = [None] * 29, [None] * 29, [None] * 29
score_exp_n, far_n, frr_n = [None] * 29, [None] * 29, [None] * 29
score_exp_o, far_o, frr_o = [None] * 29, [None] * 29, [None] * 29

with open('features-copy.csv', mode='r') as feature_file:
    feature_reader = csv.reader(feature_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    i = 0
    for (fs) in feature_reader:
        if i > 0:

            print(f"user n#{i}")
            print(fs)
            fs.pop()

            training_list, testing_dict, training_fs_list, validation_fs_dict = load_dataset(i)

            training = training_list[0:4]
            score_exp1[i - 1], far1[i-1], frr1[i-1] = test_evaluation(training, fs, testing_dict)

            training = training_list[4:8]
            score_exp2[i - 1], far2[i-1], frr2[i-1] = test_evaluation(training, fs, testing_dict)

            training = training_list[8:12]
            score_exp3[i - 1], far3[i-1], frr3[i-1] = test_evaluation(training, fs, testing_dict)

            training = training_list[12:16]
            score_exp4[i - 1], far4[i-1], frr4[i-1] = test_evaluation(training, fs, testing_dict)

            training = training_list[21:25]
            score_exp5[i - 1], far5[i-1], frr5[i-1] = test_evaluation(training, fs, testing_dict)

            training = training_list[36:40]
            score_exp6[i - 1], far6[i-1], frr6[i-1] = test_evaluation(training, fs, testing_dict)

            score_exp_g[i - 1], far_g[i - 1], frr_g[i - 1] = score_exp1[i - 1], far1[i-1], frr1[i-1]

            training = training_list[0:16]
            score_exp_h[i - 1], far_h[i - 1], frr_h[i - 1] = test_evaluation(training, fs, testing_dict)

            training = training_list[0:31]
            score_exp_i[i - 1], far_i[i - 1], frr_i[i - 1] = test_evaluation(training, fs, testing_dict)

            training = training_list[1:16] + training_list[21:25]
            score_exp_j[i - 1], far_j[i - 1], frr_j[i - 1] = test_evaluation(training, fs, testing_dict)

            training = training_list[4:16] + training_list[21:25]
            score_exp_k[i - 1], far_k[i - 1], frr_k[i - 1] = test_evaluation(training, fs, testing_dict)

            training = training_list[8:16] + training_list[21:25]
            score_exp_l[i - 1], far_l[i - 1], frr_l[i - 1] = test_evaluation(training, fs, testing_dict)

            training = training_list[12:16] + training_list[21:25]
            score_exp_m[i - 1], far_m[i - 1], frr_m[i - 1] = test_evaluation(training, fs, testing_dict)

            score_exp_n[i - 1], far_n[i - 1], frr_n[i - 1] = score_exp5[i - 1], far5[i - 1], frr5[i - 1]

            training = training_list[16:31]
            score_exp_o[i - 1], far_o[i - 1], frr_o[i - 1] = test_evaluation(training, fs, testing_dict)

        i += 1

    print(f"1)scores: {score_exp1}")
    print(f"2)scores: {score_exp2}")
    print(f"3)scores: {score_exp3}")
    print(f"4)scores: {score_exp4}")
    print(f"5)scores: {score_exp5}")
    print(f"6)scores: {score_exp6}")
    print(f"g)scores: {score_exp_g}")
    print(f"h)scores: {score_exp_h}")
    print(f"i)scores: {score_exp_i}")
    print(f"j)scores: {score_exp_j}")
    print(f"k)scores: {score_exp_k}")
    print(f"l)scores: {score_exp_l}")
    print(f"m)scores: {score_exp_m}")
    print(f"n)scores: {score_exp_n}")
    print(f"o)scores: {score_exp_o}")

    far1 = np.average(far1)
    far2 = np.average(far2)
    far3 = np.average(far3)
    far4 = np.average(far4)
    far5 = np.average(far5)
    far6 = np.average(far6)
    far_g = np.average(far_g)
    far_h = np.average(far_h)
    far_i = np.average(far_i)
    far_j = np.average(far_j)
    far_k = np.average(far_k)
    far_l = np.average(far_l)
    far_m = np.average(far_m)
    far_n = np.average(far_n)
    far_o = np.average(far_o)
    print("average false acceptance for experiment:")
    print(far1, far2, far3, far4, far5, far6, far_g, far_h, far_i, far_j, far_k, far_l, far_m, far_n, far_o)

    frr1 = np.average(frr1)
    frr2 = np.average(frr2)
    frr3 = np.average(frr3)
    frr4 = np.average(frr4)
    frr5 = np.average(frr5)
    frr6 = np.average(frr6)
    frr_g = np.average(frr_g)
    frr_h = np.average(frr_h)
    frr_i = np.average(frr_i)
    frr_j = np.average(frr_j)
    frr_k = np.average(frr_k)
    frr_l = np.average(frr_l)
    frr_m = np.average(frr_m)
    frr_n = np.average(frr_n)
    frr_o = np.average(frr_o)
    print("average false rejection for experiment:")
    print(frr1, frr2, frr3, frr4, frr5, frr6, frr_g, frr_h, frr_i, frr_j, frr_k, frr_l, frr_m, frr_n, frr_o)

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

"""

for i in range(1, 30):

    training_list, testing_dict, training_fs_list, validation_fs_dict = load_dataset(i)
    subset, eer = feature_selection(training_fs_list, validation_fs_dict)
    with open('featuresNP.csv', mode='a') as feature_file:
        feature_writer = csv.writer(feature_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        subset = list(subset)
        subset.append(eer)
        feature_writer.writerow(subset)
        
"""