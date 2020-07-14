import pandas as pd
import numpy as numpy
import os
import csv
from sklearn.model_selection import train_test_split
def make():
    data = pd.read_csv(('new_csv.csv'))
    summer_file_name='./data/summer'
    autumn_file_name='./data/autumn'
    winter_file_name='./data/winter'
    spring_file_name='./data/spring'

    par_id = os.listdir(spring_file_name)
    for count in par_id:
        dir_label = os.path.join(spring_file_name, count)
        df = pd.DataFrame({'img_name': [dir_label],
                    'season': 0})
        data= data.append(df)

    par_id = os.listdir(summer_file_name)
    for count in par_id:
        dir_label = os.path.join(summer_file_name, count)
        df = pd.DataFrame({'img_name': [dir_label],
                    'season': 1})
        data= data.append(df)

    par_id = os.listdir(autumn_file_name)
    for count in par_id:
        dir_label = os.path.join(autumn_file_name, count)
        df = pd.DataFrame({'img_name': [dir_label],
                    'season': 2})
        data= data.append(df)


    par_id = os.listdir(winter_file_name)
    for count in par_id:
        dir_label = os.path.join(winter_file_name, count)
        df = pd.DataFrame({'img_name': [dir_label],
                    'season': 3})
        data= data.append(df)

    return data

k=make()
print(len(k))
k=k.reset_index(drop=True)
train_data,test_data=train_test_split(k,test_size=0.3)
val_data, test_data=train_test_split(test_data,test_size=0.5)
train_data.reset_index(drop=True)
val_data.reset_index(drop=True)
test_data.reset_index(drop=True)

train_data.to_csv('train_dataset.csv')
val_data.to_csv('val_dataset.csv')
test_data.to_csv('test_dataset.csv')