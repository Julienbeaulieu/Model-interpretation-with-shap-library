
import scipy
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy as hc
from pandas.api.types import is_string_dtype, is_numeric_dtype, is_categorical_dtype

def display_all(df):
    '''
    Small helper function to allow us to disaply 1000 rows and columns. This will come in handy 
    because we are dealing with a lot of columns
    '''
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
        display(df)

def get_too_many_null_cols(df):
    '''
    Make a list of columns that have over 90% null values
    '''
    many_null_cols = [col for col in df.columns if df[col].isnull().sum() / df.shape[0] > 0.9]
    print("More than 90% null: " + str(len(many_null_cols)))
    return mansy_null_cols

def train_cats(df, max_n_cat):
    """
    if dtype is string
    or if dtype is numeric and cardinality is less than max_n_cat:
    change dtype to category
    """
    for n,c in df.items():
        if is_string_dtype(c) or is_numeric_dtype(c) and (c.nunique() !=2 and c.nunique() <= max_n_cat ): 
            df[n] = c.astype('category').cat.as_ordered()    

def apply_cats(df):
    '''
    Apply the same transformation to our test set keeping the category ordre from our training set
    '''
    for n,c in df.items():
        if (n in df_train.columns) and (df_train[n].dtype.name=='category'):
            df[n] = c.astype('category').cat.as_ordered() # same code as train_cats(df)
            df[n].cat.set_categories(df_train[n].cat.categories, ordered=True, inplace=True) # Use df_train as a template



def numericalize(df, col, name, max_n_cat):
    '''
    If the column is not numeric, AND if max_n_cat is not specified OR if the number of categories 
    in the columns is <= max_n_cat, then we replace the column by its category codes
    '''
    if not is_numeric_dtype(col) and (max_n_cat is None or len(col.cat.categories)>max_n_cat):
        df[name] = pd.Categorical(col).codes+1



def fix_missing(df, col, name, na_dict):
    '''
    If the column has null values or if we passed in a na_dict:
    Then we create a new column [name+'_na'] indicating where the NaNs were
    '''
    if is_numeric_dtype(col):
        if pd.isnull(col).sum() or (name in na_dict):
            df[name+'_na'] = pd.isnull(col)
            filler = na_dict[name] if name in na_dict else col.median()
            df[name] = col.fillna(filler)
            na_dict[name] = filler
    return na_dict


def proc_df(df, y_fld=None, na_dict=None, max_n_cat=None):   
    '''
    Apply numercalize and fix_missing.
    Also one hot encode category variables and split off our target variable
    '''
    df = df.copy()
    
    if y_fld is None:
        y = None
        y_fld = []
    else:
        if not is_numeric_dtype(df[y_fld]): df[y_fld] = pd.Categorical(df[y_fld]).codes
        y = df[y_fld].values
    df.drop(y_fld, axis=1, inplace=True)

    if na_dict is None: na_dict = {}
    else: na_dict = na_dict.copy()
    na_dict_initial = na_dict.copy()
    
    # Call fix_missing() to replace NaN values by the median, and create new NaN columns
    for n,c in df.items(): na_dict = fix_missing(df, c, n, na_dict)
    if len(na_dict_initial.keys()) > 0:
        df.drop([a + '_na' for a in list(set(na_dict.keys()) - set(na_dict_initial.keys()))], axis=1, inplace=True)
        
    # Apply numericalize() to change a column to it's category code
    for n,c in df.items(): numericalize(df, c, n, max_n_cat)
    df = pd.get_dummies(df, dummy_na=True) # get_dummie checks for everything that is still a category and OneHotEncodes

    res = [df, y, na_dict]

    return res

def column_cluster(df, figsize=(16,20)):
    '''
    Create a dendrogram based on spearman's r correlation
    '''
    corr = np.round(scipy.stats.spearmanr(df).correlation, 4)
    corr_condensed = hc.distance.squareform(1-corr)
    z = hc.linkage(corr_condensed, method='average')
    fig = plt.figure(figsize=figsize)
    dendrogram = hc.dendrogram(z, labels=df.columns, orientation='left', leaf_font_size=16)
    plt.show()