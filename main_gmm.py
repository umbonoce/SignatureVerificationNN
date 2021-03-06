import os
import numpy as np
import math
import csv
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import warnings
import random
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
warnings.filterwarnings("ignore")
np.random.seed(42)  # pseudorandomic
warnings.filterwarnings("ignore")
np.random.seed(42)  # pseudorandomic

N_USERS = 29

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
    testing_dict = {'skilled': [], 'genuine': [], 'random': []}  # dictionary of all testing dataframes [41..56]
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

    for u in range(1,30):
        if u != user:
            path_user = path + str(u) + '/'
            files = os.listdir(path_user)
            z = random.randint(10, 40)
            df = pd.read_csv(path_user + files[z], header=0, sep=' ',
                             names=['X', 'Y', 'TIMESTAMP', 'PENSUP', 'AZIMUTH', 'ALTITUDE', 'Z'])
            df = initialize_dataset(df)
            testing_dict['random'].append(df)


    return training_list, testing_dict, training_fs_list, validation_fs_dict


def initialize_dataset(df):
    del df['TIMESTAMP']
    del df['AZIMUTH']
    del df['ALTITUDE']
    df = compute_features(df)
    new_df = pd.DataFrame(data=None)
    for column in df.columns.values:
        new_df[column] = normalization(df[column].values)
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
    del(df['PENSUP'])
    return df


def feature_selection(training_set, validation_set):
    k = 0  # counter number of feature to select
    subset = set()  # empty set ("null set") so that the k = 0 (where k is the size of the subset)
    header = list(training_set[0].columns.values)
    total_features = set(header)
    last_score = [1] * 10
    while k != 9:

        best_score = 1
        best_trust = 0

        best_feature = ""
        feature_set = subset.copy()

        for f in (total_features - subset):
            feature_set.add(f)
            score, trust = evaluate_score(training_set, validation_set, list(feature_set))
            feature_set.remove(f)
            if (score < best_score) or (score == best_score and trust > best_trust):
                best_trust = trust
            score = evaluate_score(training_set, validation_set, list(feature_set))
            feature_set.remove(f)
            if score < best_score:
                best_score = score
                best_feature = f

        subset.add(best_feature)
        k += 1

        last_score[k] = best_score

        print("best " + str(best_feature))

        still_remove = True
        first_time = True


        while still_remove:

            if len(subset) > 1:

                feature_set = subset.copy()

                worst_feature = ""
                removed_score = 1

                for f in subset:
                    feature_set.remove(f)
                    score, trust = evaluate_score(training_set, validation_set, list(feature_set))
                    feature_set.add(f)
                    if score < removed_score:
                        removed_score = score
                        worst_feature = f

                if (worst_feature == best_feature) and first_time:
                    still_remove = False
                else:
                    if removed_score >= last_score[k-1]:  # if removed score is worse than before, go adding features
                        still_remove = False
                    else: # otherwise keep removing features
                        subset.remove(worst_feature)
                        k -= 1
                        first_time = False
                        print("removed" + str(worst_feature))
            else:
                still_remove = False

    return subset, best_score, best_trust




def evaluate_score(train_set,valid_set, features):

    print(features)
    training_set = train_set.copy()
    validation_set = valid_set.copy()
    x_list = [signature[features] for signature in training_set]
    x_train = pd.concat(x_list)

    if len(training_set) <= 15:
        n_mix = 32
    elif 15 < len(training_set) < 31:
        n_mix = 128
    else:
        n_mix = 512

    try:

        model = GaussianMixture(n_components=n_mix, random_state=42).fit(x_train)
        score_train = [None] * (len(train_set))
        score_test_gen = [None] * (len(validation_set["true"]))
        score_test_skilled = [None] * (len(validation_set["false"]))
        count_training = 0
        count_validation = 0

        for signature in x_list:
            score_train[count_training] = model.score(signature)
            count_training += 1

        average_score = np.average(score_train)
        count_training = 0

        for signature in x_list:
            distance = np.abs(score_train[count_training] - average_score)
            score_train[count_training] = np.exp((distance * -1) / len(features))
            count_training += 1

        for signature in validation_set["true"]:
            a = signature[features]
            distance = np.abs(model.score(a) - average_score)
            score_test_gen[count_validation] = np.exp((distance * -1) / len(features))
            count_validation += 1

        count_validation = 0

        for signature in validation_set["false"]:
            a = signature[features]
            distance = np.abs(model.score(a) - average_score)
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

        labels = [1] * len( validation_set["true"]) + [0] * len( validation_set["false"])
        probs = np.concatenate([score_test_gen, score_test_skilled])
        fpr, tpr, thresh = roc_curve(labels, probs)
        equal_error_rate = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        threshold = interp1d(fpr, thresh)(equal_error_rate)
        print(f"threshold: {threshold} eer: {equal_error_rate}")
    except:
        equal_error_rate = 1
        threshold = 0

        print("Fit Training Error")

    print(f"equal error rate: {equal_error_rate}")

    return equal_error_rate,threshold



def test_evaluation(train_set, features, valid_set, n_mix):

    i = 0
    training_set = train_set.copy()
    validation_set = valid_set.copy()
    x_list = [signature[features] for signature in training_set]
    x_train = pd.concat(x_list)

    try:
        model = GaussianMixture(n_components=n_mix, random_state=42).fit(x_train)
        score_train = [None] * (len(training_set))
        score_test_gen = [None] * (len(validation_set["genuine"]))
        score_test_skilled = [None] * (len(validation_set["skilled"]))

        count_training = 0

        for signature in x_list:
            score_train[count_training] = model.score(signature)
            count_training += 1

        average_score = np.average(score_train)
        count_training = 0

        for signature in x_list:
            distance = np.abs(score_train[count_training] - average_score)
            score_train[count_training] = np.exp((distance * -1) / len(features))
            count_training += 1

        count_validation = 0

        for signature in validation_set["genuine"]:
            a = signature[features]
            distance = np.abs(model.score(a) - average_score)
            score_test_gen[count_validation] = np.exp((distance * -1) / len(features))
            count_validation += 1

        count_validation = 0

        for signature in validation_set["skilled"]:
            a = signature[features]
            distance = np.abs(model.score(a) - average_score)
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

        labels = [1] * len(score_test_gen) + [0] * len(score_test_skilled)

        probs = np.concatenate([score_test_gen, score_test_skilled])
        fpr, tpr, thresh = roc_curve(labels, probs)
        equal_error_rate = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        threshold = interp1d(fpr, thresh)(equal_error_rate)
        print(f"threshold: {threshold} eer: {equal_error_rate}")
    except:
        equal_error_rate = 1
        print("Raised exception")

    return equal_error_rate


def knn_experiment(k, fs):

    train_list, testing_dict, training_fs_list, validation_fs_dict = load_dataset(k)

    training_list = [None] * len(train_list)
    count_training = 0

    for signature in train_list:
        training_list[count_training] = signature[fs]
        count_training += 1

    knn = KNeighborsClassifier(n_neighbors=5)
    labels = list()
    models = list()
    splits = list()

    splits.append(training_list[0:4])
    splits.append(training_list[4:8])
    splits.append(training_list[8:12])
    splits.append(training_list[12:16])
    splits.append(training_list[21:25])
    splits.append(training_list[36:40])

    first = np.sum(len(x) for x in training_list[0:4])
    second = np.sum(len(x) for x in training_list[4:8])
    third = np.sum(len(x) for x in training_list[8:12])
    fourth = np.sum(len(x) for x in training_list[12:16])
    fifth = np.sum(len(x) for x in training_list[21:25])
    sixth = np.sum(len(x) for x in training_list[36:40])

    x_train = pd.concat(training_list[0:4])
    models.append(GaussianMixture(n_components=32, random_state=42).fit(x_train))
    x_train = pd.concat(training_list[4:8])
    models.append(GaussianMixture(n_components=32, random_state=42).fit(x_train))
    x_train = pd.concat(training_list[8:12])
    models.append(GaussianMixture(n_components=32, random_state=42).fit(x_train))
    x_train = pd.concat(training_list[12:16])
    models.append(GaussianMixture(n_components=32, random_state=42).fit(x_train))
    x_train = pd.concat(training_list[21:25])
    models.append(GaussianMixture(n_components=32, random_state=42).fit(x_train))
    x_train = pd.concat(training_list[36:40])
    models.append(GaussianMixture(n_components=32, random_state=42).fit(x_train))

    df_training = pd.DataFrame()
    training_list = training_list[0:16] + training_list[21:25] + training_list[36:40]
    for element in training_list:
        df_training = df_training.append(element)

    labels.extend('1' for counter in range(0, first))
    labels.extend('2' for counter in range(0, second))
    labels.extend('3' for counter in range(0, third))
    labels.extend('4' for counter in range(0, fourth))
    labels.extend('5' for counter in range(0, fifth))
    labels.extend('6' for counter in range(0, sixth))

    df_labels = pd.DataFrame(data=labels, columns=['Y'])
    knn.fit(df_training, df_labels)

    average_training = [None] * len(models)
    score_test_gen = [None] * len(testing_dict["genuine"])
    score_test_skilled = [None] * len(testing_dict["skilled"])

    for m in range(0, 6):
        count_training = 0
        score_training = [None] * 4
        for signature in splits[m]:
            score_training[count_training] = models[m].score(signature)
            count_training += 1
        average_training[m] = np.average(score_training)

    i = 0
    for signature in testing_dict["genuine"]:
        a = signature[fs]
        score = knn.predict(a)
        occurrences = Counter(score)
        print(occurrences)
        index = int(max(occurrences, key=occurrences.get)) - 1
        score = models[index].score(a)
        distance = np.abs(score - average_training[index])
        score_test_gen[i] = np.exp(distance * (-1)/len(fs))
        i += 1

    i = 0
    for signature in testing_dict["skilled"]:
        a = signature[fs]
        score = knn.predict(a)
        occurrences = Counter(score)
        print(occurrences)
        index = int(max(occurrences, key=occurrences.get)) - 1
        score = models[index].score(a)
        distance = np.abs(score - average_training[index])
        score_test_skilled[i] = np.exp(distance *(-1)/len(fs))
        i += 1

    i = 0
    for score in score_test_gen:
        i += 1
        print(f" prob signature testing genuine {i}: {score}")

    i = 0
    for score in score_test_skilled:
        i += 1
        print(f" prob signature testing skilled {i}: {score}")

    labels = [1] * len(score_test_gen) + [0] * len(score_test_skilled)
    probs = np.concatenate([score_test_gen, score_test_skilled])
    fpr, tpr, thresh = roc_curve(labels, probs)
    equal_error_rate = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    threshold = interp1d(fpr, thresh)(equal_error_rate)
    print(f"threshold: {threshold} eer: {equal_error_rate}")

    return equal_error_rate


def testing():

    exp = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p']
    results = dict()
    for i in exp:
        results[i] = [None] * N_USERS

    with open('features_GMM.csv', mode='r') as feature_file:
        feature_reader = csv.reader(feature_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        i = 0
        for (fs) in feature_reader:
            if i > 0:

                print(f"user n#{i}")
                print(fs)
                fs.pop()

                training_list, testing_dict, training_fs_list, validation_fs_dict = load_dataset(i)

                n_mix = 32
                training = training_list[0:4]

                results['a'][i - 1] = test_evaluation(training, fs, testing_dict,n_mix)

                training = training_list[4:8]
                results['b'][i - 1] = test_evaluation(training, fs, testing_dict,n_mix)

                training = training_list[8:12]
                results['c'][i - 1] = test_evaluation(training, fs, testing_dict,n_mix)

                training = training_list[12:16]
                results['d'][i - 1] = test_evaluation(training, fs, testing_dict,n_mix)

                training = training_list[21:25]
                results['e'][i - 1] = test_evaluation(training, fs, testing_dict,n_mix)

                training = training_list[36:40]
                results['f'][i - 1] = test_evaluation(training, fs, testing_dict,n_mix)

                results['g'][i - 1] = results['a'][i - 1]

                n_mix = 128
                training = training_list[0:16]
                
                results['h'][i - 1] = test_evaluation(training, fs, testing_dict,n_mix)

                training = training_list[0:31]
                results['i'][i - 1] = test_evaluation(training, fs, testing_dict,n_mix)

                training = training_list[0:16] + training_list[21:25]
                results['j'][i - 1] = test_evaluation(training, fs, testing_dict,n_mix)

                training = training_list[4:16] + training_list[21:25]
                results['k'][i - 1] = test_evaluation(training, fs, testing_dict,n_mix)

                n_mix = 32
                training = training_list[8:16] + training_list[21:25]
                results['l'][i - 1] = test_evaluation(training, fs, testing_dict,n_mix)

                training = training_list[12:16] + training_list[21:25]
                results['m'][i - 1] = test_evaluation(training, fs, testing_dict,n_mix)

                results['n'][i - 1] = results['e'][i - 1]

                training = training_list[16:31]
                results['o'][i - 1] = test_evaluation(training, fs, testing_dict,n_mix)

                results['p'][i - 1] = knn_experiment(i, fs)

            i += 1

        eer = [None] * len(exp)
        i = 0
        for e in exp:
            eer[i] = np.average(results[e])*100
            i += 1

        print("average equal error rate for experiment:")
        print(eer)

        plt.plot(exp[0:6], eer[0:6], color='green', label='Ageing Experiments (a-f)')
        plt.ylabel('EER (%)')
        plt.xlabel('Experiments')
        plt.show()

        plt.plot(exp[6:15], eer[6:15], color='green', label='Ageing Experiments (a-f)')
        plt.ylabel('EER (%)')
        plt.xlabel('Experiments')
        plt.show()


def knn_testing(type):

    with open('features_GMM.csv', mode='r') as feature_file:
        feature_reader = csv.reader(feature_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        k = 0
        eer = list()
        for fs in feature_reader:
            if k > 0:
                print(f"user n#{k}")
                print(fs)
                fs.pop()

                train_list, testing_dict, training_fs_list, validation_fs_dict = load_dataset(k)

                training_list = [None] * len(train_list)
                count_training = 0

                for signature in train_list:
                    training_list[count_training] = signature[fs]
                    count_training += 1

                knn = KNeighborsClassifier(n_neighbors=5)
                labels = list()
                models = list()
                splits = list()

                splits.append(training_list[0:4])
                splits.append(training_list[4:8])
                splits.append(training_list[8:12])
                splits.append(training_list[12:16])
                splits.append(training_list[21:25])
                splits.append(training_list[36:40])

                first = np.sum(len(x) for x in training_list[0:4])
                second = np.sum(len(x) for x in training_list[4:8])
                third = np.sum(len(x) for x in training_list[8:12])
                fourth = np.sum(len(x) for x in training_list[12:16])
                fifth = np.sum(len(x) for x in training_list[21:25])
                sixth = np.sum(len(x) for x in training_list[36:40])

                x_train = pd.concat(training_list[0:4])
                models.append(GaussianMixture(n_components=32, random_state=42).fit(x_train))
                x_train = pd.concat(training_list[4:8])
                models.append(GaussianMixture(n_components=32, random_state=42).fit(x_train))
                x_train = pd.concat(training_list[8:12])
                models.append(GaussianMixture(n_components=32, random_state=42).fit(x_train))
                x_train = pd.concat(training_list[12:16])
                models.append(GaussianMixture(n_components=32, random_state=42).fit(x_train))
                x_train = pd.concat(training_list[21:25])
                models.append(GaussianMixture(n_components=32, random_state=42).fit(x_train))
                x_train = pd.concat(training_list[36:40])
                models.append(GaussianMixture(n_components=32, random_state=42).fit(x_train))

                df_training = pd.DataFrame()
                #training_list = training_list[0:16] + training_list[21:25]
                training_list = training_list[0:16] + training_list[21:25] + training_list[36:40]
                for element in training_list:
                    df_training = df_training.append(element)

                labels.extend('1' for counter in range(0, first))
                labels.extend('2' for counter in range(0, second))
                labels.extend('3' for counter in range(0, third))
                labels.extend('4' for counter in range(0, fourth))
                labels.extend('5' for counter in range(0, fifth))
                labels.extend('6' for counter in range(0, sixth))

                df_labels = pd.DataFrame(data=labels, columns=['Y'])
                knn.fit(df_training, df_labels)

                average_training = [None] * len(models)
                score_test_gen = [None] * len(testing_dict["genuine"])
                score_test_skilled = [None] * len(testing_dict[type])

                for m in range(0, len(models)):
                    count_training = 0
                    score_training = [None] * 4
                    for signature in splits[m]:
                        score_training[count_training] = models[m].score(signature)
                        count_training += 1
                    average_training[m] = np.average(score_training)

                i = 0
                for signature in testing_dict["genuine"]:
                    a = signature[fs]
                    score = knn.predict(a)
                    occurrences = Counter(score)
                    print(occurrences)
                    index = int(max(occurrences, key=occurrences.get)) - 1
                    score = models[index].score(a)
                    distance = np.abs(score - average_training[index])
                    score_test_gen[i] = np.exp(distance * (-1)/len(fs))
                    i += 1

                i = 0
                for signature in testing_dict[type]:
                    a = signature[fs]
                    score = knn.predict(a)
                    occurrences = Counter(score)
                    print(occurrences)
                    index = int(max(occurrences, key=occurrences.get)) - 1
                    score = models[index].score(a)
                    distance = np.abs(score - average_training[index])
                    score_test_skilled[i] = np.exp(distance *(-1)/len(fs))
                    i += 1

                i = 0
                for score in score_test_gen:
                    i += 1
                    print(f" prob signature testing genuine {i}: {score}")

                i = 0
                for score in score_test_skilled:
                    i += 1
                    print(f" prob signature testing {type} {i}: {score}")

                labels = [1] * len(score_test_gen) + [0] * len(score_test_skilled)
                probs = np.concatenate([score_test_gen, score_test_skilled])
                fpr, tpr, thresh = roc_curve(labels, probs)
                equal_error_rate = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
                threshold = interp1d(fpr, thresh)(equal_error_rate)
                print(f"threshold: {threshold} eer: {equal_error_rate}")
                eer.append(equal_error_rate)

            k += 1
        average_eer = np.average(eer)
        return average_eer*100


def start_fs():

    for i in range(1, 30):

        training_list, testing_dict, training_fs, validation_fs = load_dataset(i)
        subset, eer, thr = feature_selection(training_fs, validation_fs)
        with open('features_GMM_tolosana.csv', mode='a') as feature_file:
        subset, eer = feature_selection(training_fs, validation_fs)
        with open('features_GMM.csv', mode='a') as feature_file:
            feature_writer = csv.writer(feature_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            subset = list(subset)
            subset.append(eer)
            feature_writer.writerow(subset)


#eer_1 = knn_testing("skilled")
#eer_2 = knn_testing("random")

testing()
