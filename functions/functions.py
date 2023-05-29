import pandas as pd
import itertools
import numpy as np
import matplotlib.pyplot as plt
from random import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
def read_data(directory, filename):
    df = pd.read_csv(directory + filename)  # read from specified file name
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
def make_boxplot(df,columns,axis):
    df=df
    a,b = axis
    i=1
    for col in columns:
        plt.boxplot(df[col])
        plt.ylabel(col)
        plt.subplot(a, b, i)
        i+=1
    plt.show()
def make_hist(df,columns,axis):
    df=df
    a, b=axis
    i=1
    for col in columns:
        plt.hist(df[col])
        plt.xlabel(col)
        plt.subplot(a,b,i)
        i+=1
    plt.show()
def plot(x,y,xlabel,ylabel,title):
    plt.plot(x,y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
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
def write_df(df,location, filename):
    df=df
    df.to_csv(location+filename)
def get_columns(df):
    columns = [col for col in df.columns]
    return columns
def to_categorical(df, columns):
    for col in columns:
        df[col]=pd.Categorical(df[col])
        df[col]=df[col].cat.codes
    return df
def fit_rf(n_est, X_train,y_train):
    rf=RandomForestClassifier(n_estimators=n_est, random_state=42)
    rf.fit(X_train, y_train)
    return rf
def predict_rf(rf_clf, X_test):
    y_hat = rf_clf.predict(X_test)
    return y_hat
def split_data(df, target, test_size):
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=[target]), df[target], test_size=test_size)
    return X_train, X_test, y_train, y_test
def get_cnf_matrix(y_test, y_hat):
    cm=confusion_matrix(y_test, y_hat)
    return cm
def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Greens):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10,10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()
def plot_ROC(y_test, y_hat):
    fal_pos_rat, tru_pos_rat = roc_curve(y_test, y_hat)[:2]
    plt.plot(fal_pos_rat, tru_pos_rat)
    plt.grid()
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.show()
def vec_translate(a, my_dict):
    return np.vectorize(my_dict.__getitem__)(a)





