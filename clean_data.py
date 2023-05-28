from functions import functions
df = functions.read_data()
def clean_df(df):
    df=df
    '''Removing duplicates, na values, and evening out the number of samples in each category'''

    df = functions.drop(df)  # drop duplicates as well
    df = functions.drop_col_cat(df, 100, 'RiskLevel', 'low risk')
    functions.print_df(df)
    print(functions.count_cat(df, 'RiskLevel'))
    '''our class distribution for the target variable in now:
       high risk  112        
       low risk   134          
       mid risk   106

       after removing 100 low risk samples
    '''
    return df
df = clean_df(df)
functions.write_df(df,r'H:\Datasets', '\clean_MaternalHealthRisk.csv')

