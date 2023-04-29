
import pandas as pd
import numpy as np


df = pd.read_csv('local_database/MM-DatasetFallacies/no_duplicates/dataset.csv', sep='\t')
# count number of different dialogue ids
print(len(np.unique(df['Dialogue ID'].values)))

# count number of samples
print(len(df))