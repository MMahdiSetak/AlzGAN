import numpy as np
import pandas as pd

data = pd.read_csv("../dataset/all-T1.csv")

disc, counts = np.unique(data['Description'], return_counts=True)

sorted_data = pd.DataFrame({'Description': disc, 'Count': counts})
sorted_data = sorted_data.sort_values(by='Count', ascending=False).reset_index(drop=True)

sorted_data.to_csv('descriptions.csv')
print(sorted_data)
