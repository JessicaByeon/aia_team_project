from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv1D, Conv2D, Flatten, MaxPooling2D, Dropout, LSTM
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

path = 'c:/dataset/dataset/'
data_a = pd.read_csv(path+'monthly.csv')
data_b = pd.read_csv(path+'price.csv')
data_c = pd.read_csv(path+'quarter.csv') 
print(data_a.shape)
print(data_b.shape)
print(data_c.shape)

