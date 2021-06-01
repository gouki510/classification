import numpy as np
import pandas as pd
import os
import glob
import random
import itertools

data_dir = '/home/mech-user/Desktop/3S/datascience/classification/data'
listofperson = sorted(glob.glob(data_dir + '/*.csv'))
datas = []
targets = []
for j,person in enumerate(listofperson[:6]):
    count = 0
    df = pd.read_csv(person)
    for i in range(len(df)):
        if len(df.loc[i:i+19]) == 20:
            count += 1
            datas.append(np.array(df.loc[i:i+19]))
        else:
            pass
    targets.append([j]*count)
target = list(itertools.chain.from_iterable(targets))
data = np.array(datas)
print(data.shape)
print(len(target))
np.save('/home/mech-user/Desktop/3S/datascience/classification/dataset'+'/train',data)
np.save('/home/mech-user/Desktop/3S/datascience/classification/dataset'+'/target',target)