# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import random

df = pd.read_csv("dataset/Iris.csv")
df = df.drop("Id", axis=1)

df = df.rename(columns= {"species" : "label"})

def train_test_split(df, test_size):
    
    if isinstance(test_size, float):
        test_size = round(len(df) * test_size)
    
    indices = df.index.tolist()
    test_indices = random.sample(population=indices, k=test_size)
    
    test_df = df.loc[test_indices]
    train_df = df.drop(test_indices)
    
    return train_df, test_df

train_df, test_df = train_test_split(df, 20)

data = train_df.values

def check_purity(data):
    label_column = data[:, -1]
    unique = np.unique(label_column)
    
    if len(unique) == 1:
        return True
    else:
        return False

def classification(data):
    label_column = data[:, -1]
    
    unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True)
    index = counts_unique_classes.argmax()
    
    classification = unique_classes[index]
    return classification

def get_potential_splits(data):
    potential_splits = {}
    _, n_columns = data.shape
    
    for column_index in range(n_columns - 1):
        potential_splits[column_index] = []
        values = data[:, column_index]
        
        unique_values = np.unique(values)
        
        for index in range(len(unique_values) - 1):
            if index != 0:
                current_value = unique_values[index]
                previous_value = unique_values[index-1]
                
                potential_split = (current_value + previous_value) / 2
                potential_splits[column_index].append(potential_split)
        
    return potential_splits

potential_splits = get_potential_splits(data)

sns.lmplot(data=train_df, x="petal_width", y="sepal_width", fit_reg=False,
           hue="label", aspect=1.5)
plt.vlines(x=potential_splits[3], ymax=4.5, ymin=0)




















