#!/usr/bin/env python
# coding: utf-8

# In[112]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot
import seaborn as sns
import warnings
import scipy.stats as stats
warnings.filterwarnings('ignore')
#machine learning models libraries
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
#Preprocessing related libraries
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
#Date related libraries
from datetime import date
import datetime


# In[33]:


# import data. Be careful for Date type, String type (factors in R).

train = pd.read_csv("train.csv", sep=',', parse_dates=['Date'],
                    dtype={'StateHoliday': str, 'SchoolHoliday':str})

test = pd.read_csv("test.csv", sep=",", index_col = 'Id', parse_dates=['Date'],
                  dtype={'StateHoliday': str, 'SchoolHoliday':str})

store = pd.read_csv("store.csv", sep=",", dtype={'StoreType': str,'Assortment': str,'PromoInterval': str})


# In[34]:


# Inspecting train data - See first rows.
print(train.head())


# In[35]:


# Inspecting train data - See last rows.

print(train.tail())


# In[36]:


#Create new columns Year and Month to be used in the analysis of seasonal effects on sales.
train['Year'] = pd.DatetimeIndex(train['Date']).year
train['Month'] = pd.DatetimeIndex(train['Date']).month


# In[37]:


#change the order of the columns
train = train[['Store', 'DayOfWeek', 'Date', 'Year', 'Month', 'Customers', 'Open',
               'Promo', 'StateHoliday', 'SchoolHoliday', 'Sales']]
list(train.columns.values)


# In[38]:


# Inspecting train data - Data types
train.dtypes


# In[39]:


# Unique values of "StateHoliday" factor
train['StateHoliday'].unique()


# In[40]:


# convert categorical data to numeric data
# Factor levels: '0' -> 0 ; 'a' -> 1; 'b' -> 2; 'c' -> 3.

train.loc[train['StateHoliday'] == '0', 'StateHoliday'] = 0
train.loc[train['StateHoliday'] == 'a', 'StateHoliday'] = 1
train.loc[train['StateHoliday'] == 'b', 'StateHoliday'] = 2
train.loc[train['StateHoliday'] == 'c', 'StateHoliday'] = 3
train['StateHoliday'] = train['StateHoliday'].astype(int, copy=False)


# In[41]:


# Check if worked
print('levels :', train['StateHoliday'].unique(), '; data type :', train['StateHoliday'].dtype)


# In[42]:


# It worked, now automatize the process.
def factor_to_integer(df, colname, start_value=0):
    while df[colname].dtype == object:
        myval = start_value # factor starts at "start_value".
        for sval in df[colname].unique():
            df.loc[df[colname] == sval, colname] = myval
            myval += 1
        df[colname] = df[colname].astype(int, copy=False)
    print('levels :', df[colname].unique(), '; data type :', df[colname].dtype)


# In[43]:


train['SchoolHoliday'].unique()


# In[44]:


factor_to_integer(train, 'SchoolHoliday')


# In[45]:


train.dtypes


# In[48]:


print(train.describe())


# In[50]:


#checking the NaN values:
train['Open'].unique()


# In[51]:


train = train[['Store', 'DayOfWeek', 'Date', 'Year', 'Month', 'Customers', 'Open',
               'Promo', 'StateHoliday', 'SchoolHoliday', 'Sales']]
list(train.columns.values)


# In[53]:


print("NANs for individual columns")
print("---------------------------")
from collections import Counter
x = {colname : train[colname].isnull().sum() for colname in train.columns}
Counter(x).most_common()


# In[59]:


#Compute pairwise correlation of columns using pandas.corr() function.

corMat = pd.DataFrame(train.loc[:, ['DayOfWeek', 'Sales', 'Month', 'Year', 'Customers', 'Promo',
                                    'StateHoliday', 'SchoolHoliday']].corr())
print(corMat)


# In[61]:


# Visualize correlation of the DataFrame using matplotlib.pcolor() function

plt.pcolor(corMat)
plt.show()


# In[62]:


#Seaborn specializes in static charts and makes making a heatmap from a Pandas DataFrame
sns.heatmap(data=corMat)
plt.show()


# In[63]:


#examine test dataset
test.shape


# In[68]:


#Change Year and Month column data types as Date type as we did for the train dataset before.
test['Year'] = pd.DatetimeIndex(test['Date']).year
test['Month'] = pd.DatetimeIndex(test['Date']).month

print(test.head())


# In[70]:


#how many closed stores are there
sum(test['Open'] == 0)


# In[71]:


#Change the column names of the test dataset to get construct same feature names with the train dataset.
test = test[['Store', 'DayOfWeek', 'Date', 'Year', 'Month', 'Open',
             'Promo', 'StateHoliday', 'SchoolHoliday']]
list(test.columns.values)


# In[72]:


#For each column, see how many missing values exist.

print("NANs for individual columns")
print("---------------------------")
from collections import Counter
x = {colname : test[colname].isnull().sum() for colname in test.columns}
Counter(x).most_common()


# In[84]:


#There are 11 missing values in Open column. Let’s have a detailed look at those:

print(test.loc[np.isnan(test['Open'])])


# In[86]:


#Check train dataset about store 622

print(train.loc[train['Store'] == 622].head())


# In[87]:


#Avoid missing any information. So deleting the rows of Store 622 which have missing Open value, 
#should be our last option. Either label it 0 or 1. As seen above, we have data for Store 622 in train dataset. 
#Therefore, let’s label missing values of Open column in test dataset as 1.

test.loc[np.isnan(test['Open']), 'Open'] = 1


# In[88]:


#Checking if there are any missing values left.
print("NANs for individual columns")
print("---------------------------")
from collections import Counter
x = {colname : test[colname].isnull().sum() for colname in test.columns}
Counter(x).most_common()


# In[89]:


#Check for the data types for test dataset
test.dtypes


# In[90]:


#fixing StateHoliday and SchoolHoliday columns type
factor_to_integer(test, 'StateHoliday')
factor_to_integer(test, 'SchoolHoliday')
test.dtypes


# In[95]:


#Because only StateHoliday 0 and 1 exist in test dataset, 
#we should consider deleting the rows in train dataset that the StateHoliday value is different than 0 or 1.
train.loc[train['StateHoliday'] > 1].shape
train = train.loc[train['StateHoliday'] < 2]


# In[98]:


# explore the store dataset
store.shape
store.head()
store.tail()


# In[99]:


#missing values in this dataset
print("NANs for individual columns")
print("---------------------------")
from collections import Counter
x = {colname : store[colname].isnull().sum() for colname in store.columns}
Counter(x).most_common()


# In[101]:


#Promo2SinceWeek, Promo2Interval, Promo2SinceYear, CompetitionOpenSinceMonth, 
#CompetitionOpenSinceYear and CompetitionDistance columns have different size of missing values. 
#We’ll first the contents, unique values in those columns then consider imputation using either 
#scikit-learn or pandas built-in commands.

store['PromoInterval'].unique()


# In[105]:


#If there is no promotion, then the corresponding columns should have zero values.
store.loc[store['Promo2'] == 0, ['Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval']] = 0
store.loc[store['Promo2'] != 0, 'Promo2SinceWeek'] = store['Promo2SinceWeek'].max() - store.loc[store['Promo2'] != 0, 'Promo2SinceWeek']
store.loc[store['Promo2'] != 0, 'Promo2SinceYear'] = store['Promo2SinceYear'].max() - store.loc[store['Promo2'] != 0, 'Promo2SinceYear']
factor_to_integer(store, 'PromoInterval', start_value=0)
store.dtypes


# In[106]:


#Change the categorical values (string type) of StoreType and Assortment columns to integers. Check the results.
factor_to_integer(store, 'StoreType')
factor_to_integer(store, 'Assortment')
store.dtypes


# In[107]:


#An overview of the data after the latest settings.
print(store.head())


# In[109]:


#Are there still missing values?
print("NANs for individual columns")
print("---------------------------")
from collections import Counter
x = {colname : store[colname].isnull().sum() for colname in store.columns}
Counter(x).most_common()


# In[115]:


#Filling the missing values with sklearn’s built-in command. Filling with the column.mean().
from sklearn.impute import SimpleImputer
imputer = SimpleImputer().fit(store)
store_imputed = imputer.transform(store)


# In[118]:


#Check the resutls

store2 = pd.DataFrame(store_imputed, columns=store.columns.values)

print("NANs for individual columns")
print("---------------------------")
from collections import Counter
x = {colname : store2[colname].isnull().sum() for colname in store2.columns}
Counter(x).most_common()


# In[119]:


store2.head()


# In[124]:


store2['CompetitionOpenSinceMonth'] = store2['CompetitionOpenSinceMonth'].max() - store2['CompetitionOpenSinceMonth']
store2['CompetitionOpenSinceYear'] = store2['CompetitionOpenSinceYear'].max() - store2['CompetitionOpenSinceYear']
store2.tail()


# In[125]:


#Are the Store column similar in both train and store datasets?

len(store2['Store']) - sum(store2['Store'].isin(train['Store']))


# In[126]:


# Checking if there are additional (unnecessary) stores in "train" data.
# No difference at all!

StoreStore = pd.Series(store2['Store']); StoreTrain = pd.Series(train['Store'])

sum(StoreTrain.isin(StoreStore) == False)


# In[128]:


#Merge train and store datasets before modeling the data.
train_store = pd.merge(train, store2, how = 'left', on='Store')

print(train_store.head())


# In[130]:


#Check the results.
print(train_store.head())
print(train_store.tail())


# In[133]:


#Merge test and store datasets and check the result.
test_store = test.reset_index().merge(store2, how = 'left', on='Store').set_index('Id')
print(test_store.head())
test_store.shape
test_store.isnull().sum()


# In[134]:


#Visual Exploration
train_store.boxplot(column='Sales', by='Year')
plt.show()


# In[136]:


#Sales by Year
train_store.boxplot(column='Sales', by='Month')
plt.show()


# In[137]:


# Sales by month
train_store.boxplot(column='Sales', by='StateHoliday')
plt.show()


# In[138]:


#Sales in state holidays
train_store.boxplot(column='Sales', by='SchoolHoliday')
plt.show()


# In[139]:


#Sales in school holidays
train_store.boxplot(column='Sales', by='StoreType')
plt.show()


# In[141]:


# Sales by store type
train_store.boxplot(column='Sales', by='DayOfWeek')
plt.show()


# In[142]:


#Sales change through the week

train_store.boxplot(column='Sales', by='Promo2')
plt.show()


# In[151]:


# Promo and Sales change

train_store.boxplot(column='Customers', by='Month')
plt.show()


# In[154]:


#Promo interval vs sales

train_store.hist(column='Sales', by='Year', bins=30)
plt.show()


# In[155]:


#Sales & years
train_store.hist(column='Sales', by='Month', bins=30)
plt.show()


# In[156]:


#Sales & Months
train_store.hist(column='CompetitionDistance', bins=30)
plt.show()

