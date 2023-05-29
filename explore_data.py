from functions import functions

df=functions.read_data('H:\\Datasets\\', 'clean_MaternalHealthRisk.csv') # load cleaned data
columns = functions.get_columns(df) #get columns of csv file
columns.remove('RiskLevel')
print(columns)
functions.to_categorical(df,['RiskLevel']) #change risk level to categorical and assign the category codes instead of strings
functions.plot_scat(df,columns, (2,3)) # make scatter plot of RiskLevel against all feature variables in column
functions.make_boxplot(df,columns,(2,3)) # make boxplot plot of RiskLevel against all feature variables in column
functions.make_hist(df, columns, (2,3)) #make histogram plot of RiskLevel against all feature variables in column
