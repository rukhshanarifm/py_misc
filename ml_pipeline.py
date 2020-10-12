import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import datetime

def read_data(data):
    '''
    Reads data from a fle as a pandas dataframe
    
    Inputs:
    data (str): filename (including extension) for file to be read
    
    Outputs:
    df (dataframe): dataframe
    '''

    df = pd.read_csv(data)
    return df

def explore(df, variables=None, plot=False):
    '''
    Performs basic data exploration (will be updating this function for future assignments)
    Inputs:
    df (dataframe): dataframe to explore
    variables (list): initialized to None. If list given, perform operations for variables 
    in list. Otherwise, it will perform operations for all variables in the dataframe. 
    plot (boolean): boolean to identify if a plot is required
    '''

    print(df.shape)

    if variables:
        for var in variables:
            print("            ")
            print("\033[1;31;48mVariable Name: ", var)
            print("\033[0;30;48m   ")
            print(df[var].describe())
            if plot:
                df[var].value_counts().sort_values().plot(kind = 'barh')
                plt.show()
    else:
        print(df.info())
        print(df.describe())
        print("Unique Values")
        for i in df.columns:
            print(i, ": ", df[i].nunique())

    return None

def bool_to_cat(df, var):
    '''
    Converts boolean variable to a categorical one -- new variable created
    Inputs:
    df (dataframe): dataframe requiring operation
    var (str): string for variable name that requires conversion

    Output:
    df (dataframe): dataframe with new categorical variable
    '''
    var_cat = var + "_cat"
    df[var_cat] = np.where(df[var]==True, 1, 0)
    return df

def fill_with_mean(df, var_list):
    '''
    Fills missing values with mean for a list of columns

    Inputs:
    df (dataframe): input dataframe
    var_list (list): list of variables for which missing values need to be replaced

    Ouput:
    df (dataframe): updated dataframe
    '''

    assert type(var_list) == list
    for var in var_list:
        df[var] = df[var].fillna(df[var].mean())
    return df

def train_test(df, outcome_var, features, test_prop=None, seed=None):
    '''
    Splits data into training and testing

    Inputs:
    df (dataframe): input data frame
    outcome_var (str): outcome/target variable name
    features (list): list of feature names
    test_prop (float): split proportion (test/(train+test))
    seed (int): seed

    Outputs:
    Training and testing dataframes for outcome/target and features.
    '''

    assert type(outcome_var) == str
    outcome_df = df[[outcome_var]]
    features_df = df[features]

    split = train_test_split(outcome_df, features_df, test_size=test_prop, random_state=seed)
    features_train = split[0]
    features_test = split[1]
    outcome_train = split[2]
    outcome_test = split[3]

    return outcome_train, outcome_test, features_train, features_test
    

def normalize_comp(train_df, columns, test_df=None, test=False):

    '''
    Purpose: To normalize features of training/testing dataframes

    Inputs:
    train_df (dataframe): training dataframe
    column (string): feature requiring normalization
    test_df (dataframe): testing dataframe
    test (boolean): boolean to indicate if we are normalizing for testing

    Outputs:
    normalized_df (dataframe): dataframe with normalized column
    '''

    assert type(columns) == list

    if not test:
        for column in columns:
            col_norm = column + '_norm'
            train_df[col_norm] = (train_df[column] - train_df[column].mean()) / train_df[column].std()
            ret_df = train_df
    else:
        for column in columns:
            col_norm = column + '_norm'
            train_mean = train_df[column].mean()
            train_std = train_df[column].std()
            test_df[col_norm] = (test_df[column] - train_mean) / train_std
            ret_df = test_df

    return ret_df

        
def neg_to_miss(df, var_list):
    '''
    Converts negative values to misisng (if required)

    Inputs:
    df (dataframe): input dataframe
    var_list (list): list of variables

    Ouputs:
    df (dataframe): updated dataframe
    '''

    for var in var_list:
        df[var] = np.where(df[var] < 0 ,np.NaN, df[var])

    return df

def check_if_norm(df, features):
    '''
    Prints out feature summaries to check if features have been normalized
    
    Inputs:
    df (dataframe): input dataframe
    features (list): list of features

    Output:
    None
    '''

    for feature in features:
        print(feature)
        feature = feature + "_norm"
        print("Mean: ", df[feature].mean())
        print("Standard Deviation ", df[feature].std())

    return None

def create_dummies(features_train, features_test, features):
    '''
    Creates dummies and performs one-hot-encoding in training/testing variables
    when required (This function requires a bit of cleaning for future use)

    Inputs:
    features_train (dataframe): dataframe of training features
    features_test (dataframe): dataframe of testing features
    features (list): list of features

    Ouputs:
    Updated dataframes
    '''

    #unique values in training
    for feature in features:

        #List of unique values in training and testing dataframes
        #for a given feature
        unique = features_train[feature].drop_duplicates().to_list()
        unique_test = features_test[feature].drop_duplicates().to_list()

        #Check if each value in training is in the testing values
        for each_categ in unique:
            if each_categ in unique_test or not each_categ:
                pass
            else:
                each_categ = feature + "_" + str(each_categ)
                features_test[each_categ] = 0
        dummy = pd.get_dummies(features_train[feature], prefix=feature)
        features_train = pd.concat([features_train, dummy], axis=1)

        #Check if each value in testing is in the training values
        for each_categ in unique_test:

            if each_categ in unique:
                pass
            else:
                features_test.drop(features_test[features_test[feature] == each_categ].index, inplace = True)

        dummy = pd.get_dummies(features_test[feature], prefix=feature)
        features_test = pd.concat([features_test, dummy], axis=1)

    return features_train, features_test


def grid_search(GRID, MODELS, features_train, features_test, outcome_train, outcome_test):
    '''
    Runs a set of models multiple times (each with different parameters) and
    computes accuracy score.

    Inputs:
    GRID (dict): dictionary of models and parameters
    MODELS (dict): dictionary of model names
    features_train (dataframe): features training df
    features_test (dataframe): features testing df
    outcome_train (dataframe): outcome/target training df
    outcome_test (dataframe): outcome/taarget testing df

    Outputs:
    df (dataframe): dataframe with va
    coef_list (list): list of coefficients for each model (not characterized or labelled)
    sum_df (dataframe): summary dataframe
    '''

    # Begin timer 
    start = datetime.datetime.now()

    # Initialize results data frame 
    sum_df = pd.DataFrame()
    loop_df = pd.DataFrame()

    coef_list = []

    # Loop over models 
    for model_key in MODELS.keys():

        # Loop over parameters 
        for params in GRID[model_key]:

            print(model_key)
            loop_df['Model'] = model_key
            loop_df['parameters'] = [params]
            print("Training model:", model_key, "|", params)

            # Create model 
            model = MODELS[model_key]
            model.set_params(**params)

            # Fit model on training set 
            model.fit(features_train, outcome_train)
            if model_key != "GaussianNB":
                coef_list.append(model.coef_)

            # Predict on testing set 
            target_pred = model.predict(features_test)

            # Evaluating
            loop_df['accuracy'] = accuracy_score(outcome_test[['Arrest_cat']], target_pred)
            sum_df = sum_df.append(loop_df)

    # End timer
    stop = datetime.datetime.now()
    print("Time Elapsed:", stop - start)

    return coef_list, sum_df
    