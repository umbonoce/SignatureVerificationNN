import os
import pandas as pd
import matplotlib.pyplot as pyp

path = './xLongSignDB/'
path_user = path + str(7) + '/'
files = os.listdir(path_user)
i = 0

for file in files:
    df = pd.read_csv(path_user + file, header=0, sep=' ',
                     names=['X', 'Y', 'TIMESTAMP', 'PENSUP', 'AZIMUTH', 'ALTITUDE', 'Z'])
    df = df[['X','Y']]

    i += 1

    if 'ss' in file:
        if i == 50:
            pyp.plot(df['X'], df['Y'])
            pyp.show()
    if 'sg' in file:
        if i == 1 or i == 5 or i == 9 or i == 13 or i == 23 or i == 37:
            pyp.plot(df['X'], df['Y'])
            pyp.show()