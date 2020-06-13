import pandas as pd
import numpy as numpy
from sklearn.model_selection import train_test_split
def make(phase):

    data = pd.read_csv(('new_csv.csv'))

    train_data,test_data=train_test_split(data,test_size=0.3)
    val_data, test_data=train_test_split(test_data,test_size=0.5)
    if phase=='train':
        return train_data.reset_index(drop=True)
    elif phase=='test':
        return test_data.reset_index(drop=True)
    elif phase=='val':
        return val_data.reset_index(drop=True)

