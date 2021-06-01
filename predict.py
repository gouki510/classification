import numpy as np 
import pandas as pd
import os
import glob
import torch
from natsort import natsorted
from train import LSTM


path2data = '/home/mech-user/Desktop/3S/datascience/classification/data'
listofdata = natsorted(glob.glob(path2data+"/*csv"))
results = []
for data in listofdata[6:]:
    df = pd.read_csv(data)
    data = np.array(df)
    data = torch.from_numpy(data)
    data = data.view(1,20,6)
    model = LSTM()
    model.load_state_dict(torch.load("model/lstm.pt"))
    model.eval()
    output = model(data.float())
    result = output.max(1)[1].item()
    if result == 0:
        results.append('a')
    elif result == 1:
        results.append('b')
    elif result == 2:
        results.append('c')
    elif result == 3:
        results.append('d')
    elif result == 4:
        results.append('e')
    elif result == 5:
        results.append('f')

predict = pd.DataFrame(results)
predict = predict.T
predict.to_csv('/home/mech-user/Desktop/3S/datascience/classification/predict.csv')

