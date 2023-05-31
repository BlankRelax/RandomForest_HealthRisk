from functions import functions

df = functions.read_data(
    'H:\\Datasets\\',
    'clean_MaternalHealthRisk.csv')  # load cleaned data
df = df.drop(columns=['BodyTemp'])
X_train, X_test, y_train, y_test = functions.split_data(
    df, 'RiskLevel', 0.2)  # split dataset
print('len of train: ', len(X_train), 'len of test: ', len(X_test))

clf = functions.fit_bdt(X_train, y_train)  # fit data
functions.predict_bdt(clf, X_test, y_test)  # predict data and print metrics
