import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

test_df = pd.read_csv('./file/test.csv')
train_df = pd.read_csv('./file/train.csv')
print(test_df.head())
print(train_df.head())