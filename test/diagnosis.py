import numpy as np
import pandas as pd
from tqdm import tqdm

dgs = pd.read_csv("DXSUM.csv")
dgs = dgs.dropna(subset=['DIAGNOSIS'])
subjects = np.unique(dgs['PTID'])
un, num = np.unique(dgs['DIAGNOSIS'], return_counts=True)
change = [0, 0, 0]

for sbj in tqdm(subjects):
    gp = np.unique(dgs.loc[dgs['PTID'] == sbj]['DIAGNOSIS'])
    change[len(gp) - 1] += 1

print(change)
