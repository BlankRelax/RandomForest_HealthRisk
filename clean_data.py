from functions import functions
df = functions.read_data(r'H:\Datasets', '\\Maternal Health Risk Data Set.csv')


def clean_df(df):
    df = df
    '''Removing duplicates, na values, and evening out the number of samples in each category'''

    df = functions.drop(df)  # drop duplicates as well
    df = functions.drop_col_cat(
        df,
        140,
        'RiskLevel',
        'low risk')  # drop 140 low risk samples
    functions.print_df(df)
    # count the number of samples from each class to make sure it has worked
    print(functions.count_cat(df, 'RiskLevel'))
    '''our class distribution for the target variable in now:
       high risk  112
       low risk   94
       mid risk   106

       after removing 140 low risk samples
       we made the low risk samples significantly less than the rest, because we want to maximise accuracy on high risk patients
    '''
    return df


df = clean_df(df)  # call function

# expect error if we save a file with an existing file name
try:
    # save cleaned data to csv fileSkl
    functions.write_df(df, r'H:\Datasets', '\\clean_MaternalHealthRisk.csv')
except PermissionError:
    print('chosen file name already exists, choose another one')
