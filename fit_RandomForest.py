from functions import functions
import numpy as np

df=functions.read_data('H:\\Datasets\\', 'clean_MaternalHealthRisk.csv') # load cleaned data
df=df.drop(columns=['BodyTemp'])
X_train, X_test, y_train, y_test=functions.split_data(df,'RiskLevel',0.2) #split dataset
print('len of train: ',len(X_train),'len of test: ', len(X_test))


n_est_list = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
#functions.rf_param_search(X_train, y_train, X_test, y_test, n_est_list)
functions.get_rf_cm(100, X_train, y_train,X_test, y_test )





