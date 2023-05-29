from functions import functions
import numpy as np

df=functions.read_data('H:\\Datasets\\', 'clean_MaternalHealthRisk.csv') # load cleaned data
X_train, X_test, y_train, y_test=functions.split_data(df,'RiskLevel',0.2) #split dataset

print('len of train: ',len(X_train),'len of test: ', len(X_test))


n_est_list= [1,5,10,15,20,25,30,35,40,45,50,55,60,65,70]
rel_error_array=[]
for n_est in n_est_list:
    rel_error_array_temp = []
    for i in range(50):
        rf = functions.fit_rf(n_est, X_train, y_train)
        y_hat = functions.predict_rf(rf, X_test)
        rel_error = np.array(y_test == y_hat).sum() / np.array(y_test).shape[0]
        rel_error_array_temp.append(rel_error)
    rel_error_array.append(np.mean(rel_error_array_temp))
functions.plot(n_est_list,rel_error_array,'n_est','relative error','Avg relative error against n_est')






