from functions import functions
df = functions.read_data(
    'H:\\Datasets\\',
    'clean_MaternalHealthRisk.csv')  # load cleaned data
df = df.drop(columns=['BodyTemp'])
X_train, X_test, y_train, y_test = functions.split_data(
    df, 'RiskLevel', 0.2)  # split dataset
print('len of train: ', len(X_train), 'len of test: ', len(X_test))
# clf=functions.fit_dt('gini', X_train, y_train)
# functions.predict_dt(clf,X_test, y_test)
crits = ['gini', 'entropy', 'log_loss']
splitters = ['best', 'random']
for crit in crits:
    clf = functions.fit_dt(crit, 'best', X_train, y_train)
    functions.predict_dt(clf, X_test, y_test)
