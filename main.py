import numpy as np
import math
import os
import pandas as pd
import sklearn
from hmmlearn import hmm
from sklearn.model_selection import KFold
np.random.seed(42)  # pseudorandomic


# tanh normalization; values between 0 and 1
def normalization(data):
    mean = data.mean()
    std = data.std() + np.finfo(np.float32).eps
    return 0.5 * (np.tanh(0.01 * ((data - mean) / std)) + 1)


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


def load_dataset(users):
    df_dict = {user: {'skilled': [], 'genuine': []} for user in users}  # dictionary of path names

    for user in users:
        path = './xLongSignDB/' + user
        files = os.listdir(path)

        for file in files:
            if 'ss' in file:
                df_dict[user]['skilled'].append(pd.read_csv(path + "/" + file, header=0, skiprows=1, sep=' ',
                                                            names=['X', 'Y', 'TIMESTAMP', 'PENSUP', 'AZIMUTH',
                                                                   'ALTITUDE',
                                                                   'Z']))

            elif 'sg' in file:
                df_dict[user]['genuine'].append(pd.read_csv(path + "/" + file, header=0, skiprows=1, sep=' ',
                                                            names=['X', 'Y', 'TIMESTAMP', 'PENSUP', 'AZIMUTH',
                                                                   'ALTITUDE', 'Z']))
            else:
                print(file)

    return df_dict


def initialize_dataset(sign_dict):
    for user in sign_dict:
        for df in sign_dict[user]['genuine']:
            del df['TIMESTAMP']
            del df['AZIMUTH']
            del df['ALTITUDE']
            df['X'] = normalization(df['X'])
            df['Y'] = normalization(df['Y'])
            df['Z'] = normalization(df['Z'])
            df = compute_features(df)

        for df in sign_dict[user]['skilled']:
            del df['TIMESTAMP']
            del df['AZIMUTH']
            del df['ALTITUDE']
            df['X'] = normalization(df['X'])
            df['Y'] = normalization(df['Y'])
            df['Z'] = normalization(df['Z'])
            df = compute_features(df)

    return sign_dict


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


def feature_selection(x_train, y_train):
    header = list(x_train[0].columns.values)
    '''
        concatenation = x_train[0]

    for i in range(1, len(x_train)):
        concatenation = np.concatenate([concatenation, x_train[i]])

    x_dataset = pd.DataFrame(data=concatenation, columns=header)
    '''

    k = 0
    subset = set()  # empty set ("null set") so that the k = 0 (where k is the size of the subset)
    total_features = set(header)

    while k != 3:
        best_score = 0
        best_feature = ""
        copy = subset.copy()
        for f in (total_features - subset):
            copy.add(f)
            score = evaluate_score(x_train,list(copy), y_train)
            copy.remove(f)
            if score > best_score:
                best_score = score
                best_feature = f

        subset.add(best_feature)
        worst_score = 1
        worst_feature = ""
        copy = subset.copy()

        if len(subset) != 1:
            for f in subset:
                copy.remove(f)
                score = evaluate_score(x_train,list(copy), y_train)
                copy.add(f)
                if score < worst_score:
                    worst_score = score
                    worst_feature = f

            if worst_feature == best_feature:
                k += 1
            else:
                subset.remove(worst_feature)
        else:
            k += 1

    return subset




def evaluate_score(x_dataset,features,y_dataset):

    kf = KFold(n_splits=6,shuffle=True)
    x_test = [None] * 7
    x_train = [None] * (len(x_dataset)-len(x_test))
    array = np.array(x_dataset)

    for train_index, test_index in kf.split(x_dataset):
        print("train: "+ str(train_index) +" test: "+str(test_index))
        x_train, x_test = array[train_index], array[test_index]

    print(len(x_train))
    print(len(x_test))
    return np.random.rand()


user_folders = os.listdir('./xLongSignDB')
df_dict = load_dataset(user_folders)
# df_dict = initialize_dataset(df_dict)

for user in df_dict:
    x_train_set = [None] * 41
    y_train_set = [None] * 41
    i = 0
    for df in df_dict[user]['genuine']:
        x_train_set[i] = df
        y_train_set[i] = len(df)
        i = i + 1
        if (i == 41):
            break

    # x_train_set = pd.DataFrame(data=x_train_set, columns=header)
    # y_train_set = pd.DataFrame(data=y_train_set,columns=['Label'])
    print(x_train_set)
    feature_selection(x_train_set, y_train_set)
