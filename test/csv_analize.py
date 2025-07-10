import numpy as np
import pandas as pd
from tqdm import tqdm

data = pd.read_csv("../dataset/all-T1.csv")
for sbj in tqdm(data['Subject ID'].values):
    gp = np.unique(data.loc[data['Subject ID'] == sbj]['Research Group'])
    if len(gp) > 1:
        print(sbj)

# disc, counts = np.unique(data['Description'], return_counts=True)
#
# sorted_data = pd.DataFrame({'Description': disc, 'Count': counts})
# sorted_data = sorted_data.sort_values(by='Count', ascending=False).reset_index(drop=True)
#
# sorted_data.to_csv('descriptions.csv')
# print(sorted_data)
