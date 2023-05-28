from functions import functions

df=functions.read_data('H:\\Datasets\\', 'clean_MaternalHealthRisk.csv')
columns = functions.get_columns(df)
functions.to_categorical(df,['RiskLevel'])
functions.plot_scat(df,columns, (2,3))
functions.make_boxplot(df,(3,3))