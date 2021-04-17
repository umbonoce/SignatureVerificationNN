import numpy as np
import os
import pandas as pd
from hmmlearn import hmm

import matplotlib.pyplot as plt


# tanh normalization; values between 0 and 1
def normalization(data):
    mean = data.mean()
    std = data.std() + np.finfo(np.float32).eps
    return 0.5 * (np.tanh(0.01 * ((data - mean) / std)) + 1)


def load_dataset(users):

    df_dict = {user: {'skilled': [], 'genuine': []} for user in users}  # dictionary of path names

    for user in users:
        path = './xLongSignDB/' + user
        files = os.listdir(path)

        for file in files:
            if 'ss' in file:
                df_dict[user]['skilled'].append(pd.read_csv(path + "/" + file, header=0, skiprows=1, sep=' ',
                                                            names=['X', 'Y', 'TIMESTAMP', 'PENISUP', 'AZIMUTH',
                                                                   'ALTITUDE',
                                                                   'Z']))

            elif 'sg' in file:
                df_dict[user]['genuine'].append(pd.read_csv(path + "/" + file, header=0, skiprows=1, sep=' ',
                                                            names=['X', 'Y', 'TIMESTAMP', 'PENISUP', 'AZIMUTH',
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
            compute_features(df)
        for df in sign_dict[user]['skilled']:
            del df['TIMESTAMP']
            del df['AZIMUTH']
            del df['ALTITUDE']
            df['X'] = normalization(df['X'])
            df['Y'] = normalization(df['Y'])
            df['Z'] = normalization(df['Z'])
            compute_features(df)
    return


def compute_features(df):
    new_df = df.copy()
    e = np.finfo(np.float32).eps
    leng = len(df)

    ''' derivate feature '''
    dv_list = np.array([])
    da_list = np.array([])
    dp_list = np.array([])
    dtheta_list = np.array([])

    x_list = df['X'].tolist()
    y_list = df['Y'].tolist()
    z_list = df['Z'].tolist()  # pressione

    dx_list = second_order_regression(x_list)
    dy_list = second_order_regression(y_list)

    theta_list = [None] * leng

    for i in range(0, leng):
        theta_list[i] = np.arctan(dy_list[i]/(dx_list[i]+e))

    print(theta_list)

    dz_list = second_order_regression(z_list)

    v_list = np.array([])  # velocità
    p_list = np.array([])  # ro
    a_list = np.array([])  # accelerazione





    ddx_list = np.array([])
    ddy_list = np.array([])

    ratios_list = np.array([])  # Ratio of the minimum over the maximum speed over a 5-samples window
    alphacons_list = np.array([])  # Angle of consecutive samples
    dalpha_list = np.array([])
    sinalpha_list = np.array([])
    cosalpha_list = np.array([])
    strokeratio5_list = np.array([])  # Stroke length to width ratio over a 5-samples window
    strokeratio7_list = np.array([])  # Stroke length to width ratio over a 7-samples window



    return new_df.copy()


def second_order_regression(data_list):
    pad_list = data_list.copy()
    pad_list.append(data_list[len(data_list)-1])
    pad_list.append(data_list[len(data_list)-1])
    pad_list.insert(0, data_list[0])
    pad_list.insert(0, data_list[0])
    derivata = [None] * (len(data_list))

    for n in range(2, len(data_list)+2):
        derivata[n-2] = ((pad_list[n+1]-pad_list[n-1]+2*(pad_list[n+2]-pad_list[n-2]))/10)

    return derivata

np.random.seed(42)

users = os.listdir('./xLongSignDB')
df_dict = load_dataset(users)

initialize_dataset(df_dict)
