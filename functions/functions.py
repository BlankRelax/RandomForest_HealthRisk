import pandas as pd
import matplotlib.pyplot as plt
from random import randint

def read_data():
    directory = r'H:\Datasets'  # change here for directory
    df = pd.read_csv(directory + '\Maternal Health Risk Data Set.csv')  # read from specified file name
    columns = [col for col in df.columns]
    columns
    return df

def print_df(df,head='none'): # head arg gives user option to print whole or head of df
    option=head
    if option=='true':
        print(df.head())
    else:
        print(df)
def drop(df):
    df=df.dropna()
    df=df.drop_duplicates()
    return df
def plot_scat(df,x, axis): # takes feature list as input
    a,b=axis
    df=df
    for i in range(len(x)-1):
        plt.scatter(df[x[i]], df['RiskLevel'])
        plt.ylabel('Risk Level(target)')
        plt.xlabel(x[i])
        plt.subplot(a,b,i+1)
    plt.show()
def count_cat(df,column): #counts categories in a categorical column
    df=df
    return df.groupby(column).count()
def drop_col_cat(df,num, column,category):
    num_dropped = 0
    df=df
    while num_dropped < num:
        ran = randint(0, 1013) # this range will work for all columns
        try:
            df[df[column] == category] = df[df[column] == category].drop(ran)
            num_dropped += 1
        except KeyError:
            pass
    df.merge(df[df['RiskLevel'] == 'low risk'])
    df=df.dropna()
    return df




##########################data exploration##################################


#plot_scat(df,columns,(3,2))


#print(count_cat(df,'RiskLevel'))
'''
We have:
high risk  272         
low risk   406         
mid risk   336

lets remove the number of low risk samples, as we want to focus on high risk classification as this is more important.
lets reduce low risk from 406 -> 320
'''



#df=drop_col_cat(df,86, 'RiskLevel', 'low risk')
#print(count_cat(df,'RiskLevel'))

'''Now we have a more balanced category distribution:
   high risk  272        
   low risk   320         
   mid risk   336

'''




