from functions import functions
import pandas as pd
import matplotlib.pyplot as plt
from random import randint

'''Removing duplicates, na values, and evening out the number of samples in each category'''
df = functions.read_data()
df=functions.drop(df)
df=functions.drop_col_cat(df,100,'RiskLevel','low risk')
functions.print_df(df)
print(functions.count_cat(df,'RiskLevel'))
'''our class distribution for the target variable in now:
   high risk  112        
   low risk   134          
   mid risk   106
   
   after removing 100 low risk samples
'''

