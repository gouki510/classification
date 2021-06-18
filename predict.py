import numpy as np 
import pandas as pd
import os
import glob
import torch
from natsort import natsorted
from train import LSTM
from main import CNN


path2data = '/home/mech-user/Desktop/3S/datascience/classification/data'
listofdata = natsorted(glob.glob(path2data+"/*csv"))
results = []
for data in listofdata[6:]:
    df = pd.read_csv(data)
    data = np.array(df)
    data = torch.from_numpy(data)
    data = data.unsqueeze(0)
    """
    data = torch.transpose(data,0,2)
    data = torch.transpose(data,1,2)
    data = data.unsqueeze(0)"""
    # model change
    model = CNN()
    model.load_state_dict(torch.load("model/cnn.pt"))
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
predict.to_csv('/home/mech-user/Desktop/3S/datascience/classification/predict_cnn.csv')

