'''
Created on Dec 10, 2012
@author: Stephen O'Hara

Utility functions for data management. These are the
types of functions used to normalize values, etc.

For loading/saving data of various formats, please refer to the
data_io module.
'''
import scipy as sp

def constant_cols(D):
    '''
    Generate an array that lists those columns that have
    zero standard deviation in the input matrix.
    @param D: The input data matrix, features in columns
    @return: A list of those columns which have zero std dev
    '''
    s = D.std(axis=0)
    idxs = sorted( [ i for i in range(len(s)) if s[i]==0])
    return idxs

def indicator_vars(L):
    '''
    Transforms the list of unique class labels, L, into
    an indicator matrix. One row in the matrix for each label.
    One column for each unique class (alphabetically sorted),
    such that a 1 in the column indicates the class
    @return: tuple (IV, unique_labels) where IV is the indicator
    variable matrix and unique_labels is the ordered set of 
    labels that correspond with the indicator columns.
    '''
    unique_labels = sorted( list(set(L)) )
    P = len(unique_labels)
    N = len(L)
    IV = sp.zeros((N,P))
    for i in range(N):
        tmp_label = L[i]
        iv_column = unique_labels.index(tmp_label)
        IV[i,iv_column] = 1
        
    return IV.astype(int), unique_labels

def normalize_data(D, means=None, stdvs=None):
    '''
    Transforms data matrix D such that columns (features) are mean-centered
    and unit-deviation. Remember to transform your testing data using
    the means and stdvs computed from your training data!!!
    @param D: NxP data matrix, rows are samples, features are columns
    @param means: An array of P values, representing the mean value that
    should be subtracted from each feature. Default is None, indicating the
    means should be computed from the samples in D.
    @param stdvs: An array of P values, representing the std deviations of
    the P columns to be used to normalize the columns. Default is None, indicating
    the stdevs will be computed from the samples in D.
    @return (Dn, means, stdvs) where Dn is NxP, such that columns are mean-centered
    and have unit deviation. means is the list of P mean values, and stdvs is the 
    list of standard deviations for each column
    '''
    means = sp.mean(D,axis=0) if means is None else means
    stdvs = sp.std(D,axis=0) if stdvs is None else stdvs
    Dn = (D - means) / stdvs
    return (Dn, means, stdvs)


def replace_missing_values_with_col_means(D, sentinel=-100000):
    '''
    For a numeric data matrix, D, cells which have missing
    data will be replaces with the column average for the cell's column.
    @param D: A numpy/scipy array. Assumes rows are samples and columns are field values
    @param sentinel: This is the special number that denotes a missing value. This
    must have been added to the data matrix by another function.
    @return: Dhat, where missing values are now mean values for the column.
    '''
    D2 = D.copy()
    _rows, cols = D.shape
    for c in range(cols):
        S = D[:,c] #array slice
        S2 = D2[:,c]
        mask1 = (S == sentinel)
        if sum( mask1 ) > 0:
            mask2 = (S != sentinel)
            S2[mask1] = sp.mean( S[mask2] )

    return D2
    
def replace_missing_values_with_value(D, sentinel=-100000, val=0):
    '''
    For a numeric data matrix, D, cells which have missing
    data will be replaces with the constant value specified
    in the arguments.
    @param D: A numpy/scipy array. Assumes rows are samples and columns are field values
    @param sentinel: This is the special number that denotes a missing value. This
    must have been added to the data matrix by another function.
    @param val: This is the number to use in place of missing entries
    @return: Dhat, where missing values are now mean values for the column.
    '''
    D2 = D.copy()
    _rows, cols = D.shape
    for c in range(cols):
        S = D[:,c] #array slice
        S2 = D2[:,c]
        mask1 = (S == sentinel)
        S2[mask1] = val

    return D2    
    
def gappy_means_and_stdvs(X, sentinel=0):
    '''
    Computes the column means and standard deviations of a data matrix X, where some of the values
    are missing or unknown. This function simply ignores those entries marked
    with the sentinel value in the computation.
    @param X: Data matrix, the columns of which will be processed, rows are assumed to be multiple
    observations.
    @param sentinel: The numeric value that represents a missing entry in X.
    @return: (M, S, N), where the mean and standard deviation of a column are computed using
    only the non-missing entries. I.e., mean(Y) = (sum Yi)/Nhat where i are the non-missing indexes, and
    Nhat is the number of non-missing entries. M is the vector of means, S is the vector of std devs,
    and N has the number of non-missing values per column
    '''
    (_N,p) = X.shape
    M = sp.zeros(p)  #vector of mean values
    S = sp.zeros(p)  #vector of std devs
    N = sp.zeros(p)  #vector of N, the number of non-missing values per column
    for col in range(p):
        v = X[:,col]
        x = v[ v != sentinel ]
        M[col] = x.mean()
        S[col] = x.std()
        N[col] = len(x)
        
    return (M,S,N)

    