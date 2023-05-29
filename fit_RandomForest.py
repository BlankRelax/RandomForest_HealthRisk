from functions import functions
import numpy as np

df=functions.read_data('H:\\Datasets\\', 'clean_MaternalHealthRisk.csv') # load cleaned data
X_train, X_test, y_train, y_test=functions.split_data(df,'RiskLevel',0.2) #split dataset

print('len of train: ',len(X_train),'len of test: ', len(X_test))


n_est_list= [1,5,10,15,20,25,30,35,40,45,50,55,60,65,70]
rel_error_array=[]
for n_est in n_est_list:
    rel_error_array_temp = []
    for i in range(2):
        rf = functions.fit_rf(n_est, X_train, y_train)
        y_hat = functions.predict_rf(rf, X_test)
        rel_error = np.array(y_test == y_hat).sum() / np.array(y_test).shape[0]
        rel_error_array_temp.append(rel_error)
    rel_error_array.append(np.mean(rel_error_array_temp))
functions.plot(n_est_list,rel_error_array,'n_est','relative error','Avg relative error against n_est')
risk_keys = {'low risk':0, 'mid risk':1,'high risk':2}
y_test = functions.vec_translate(np.array(y_test),risk_keys)
y_hat = functions.vec_translate(np.array(y_hat),risk_keys)

cnf_matrix = functions.get_cnf_matrix(y_test,y_hat)
print(cnf_matrix)
functions.plot_confusion_matrix(cnf_matrix,['high risk', 'mid risk', 'low risk'])




