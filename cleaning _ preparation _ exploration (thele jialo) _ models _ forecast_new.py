#data exploration cleaning and preparation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
#machine learning models libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

#Preprocessing related libraries
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from scipy.stats import boxcox
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# import data. Be careful for Date type, String type (factors in R).
train = pd.read_csv("train.csv", sep=',', parse_dates=['Date'], dtype={'StateHoliday': str, 'SchoolHoliday':str})
test = pd.read_csv("test.csv", sep=",", index_col = 'Id', parse_dates=['Date'], dtype={'StateHoliday': str, 'SchoolHoliday':str})
store = pd.read_csv("store.csv", sep=",", dtype={'StoreType': str,'Assortment': str,'PromoInterval': str})

#train = pd.read_csv("train.csv")
#test = pd.read_csv("test.csv")
#store = pd.read_csv("store.csv")


#we have some categorical data, we need to convert them to numerical. we can easily do it cause they are alpabetically
#sorted 'a'-> 0, 'b' -> 1, etc.

# this function converts categorical column data to numeric
def categorical_to_numerical(df, colname, start_value=0):
    while df[colname].dtype == object:
        myval = start_value # factor starts at "start_value".
        for sval in df[colname].unique():
            df.loc[df[colname] == sval, colname] = myval
            myval += 1
        df[colname] = df[colname].astype(int, copy=False)

#----------------------------------------------------------------------------------------------------------

#Train Dataset:

#inspect train data:
print(train.shape)
print(train.dtypes)
print(train.head())
print(train.tail()) 


#drop duplicate values if exist
#print(train.shape, '\n')
train = train.drop_duplicates()
#print(train.shape, '\n')
#we see that no rows were dropped, we had no duplicates

#drop closed columns where stores are closed
# drop stores that are closed in train dataset
train = train[train.Open != 0]

print("Open Stores:",sum(train['Open'] == 1))
print("Closed Stores:",sum(train['Open'] == 0))


#break date into year, month
#we did not choose day cause we already have it
train['Year'] = pd.DatetimeIndex(train['Date']).year
train['Month'] = pd.DatetimeIndex(train['Date']).month

#convert categorical to numerical 
print('levels before:', train['StateHoliday'].unique(), '; data type :', train['StateHoliday'].dtype)
categorical_to_numerical(train,'StateHoliday')
print('levels after:', train['StateHoliday'].unique(), '; data type :', train['StateHoliday'].dtype)

#it worked
#do the same to SchoolHoliday
categorical_to_numerical(train, 'SchoolHoliday')

#Check the data types for each column.
print(train.dtypes)

#now all of our columns are numeric
#get a view on the data
print(train.describe())


#check for NaNs in our selected columns
print("NANs for individual columns")
print("---------------------------")
from collections import Counter
x = {colname : train[colname].isnull().sum() for colname in train.columns}
Counter(x).most_common()

#we have no NANs in train dataset

#select columns for the training data
train = train[['Store', 'DayOfWeek', 'Date', 'Year', 'Month', 'Customers', 'Open','Promo', 'StateHoliday', 'SchoolHoliday', 'Sales']]
list(train.columns.values)

#Compute pairwise correlation of columns using pandas.corr() function.
corMat = pd.DataFrame(train.loc[:, ['DayOfWeek', 'Sales', 'Month', 'Year', 'Customers', 'Promo','StateHoliday', 'SchoolHoliday']].corr())
print(corMat)

# visualize correlations using heatmap
sns.heatmap(data=corMat)



#----------------------------------------------------------------------------------------------------------
#Test Dataset:

#inspect Test data:
print(test.shape)
print(test.dtypes)
print(test.head())
print(test.tail()) 

#drop duplicate values if exist
print(test.shape, '\n')
test = test.drop_duplicates()
print(test.shape, '\n')
#we see that no rows were dropped, we had no duplicates

#drop closed columns where stores are closed
# drop stores that are closed in train dataset
test = test[test.Open != 0]

print("Open Stores:",sum(test['Open'] == 1))
print("Closed Stores:",sum(test['Open'] == 0))


#break date into year, month
#we did not choose day cause we already have it
test['Year'] = pd.DatetimeIndex(test['Date']).year
test['Month'] = pd.DatetimeIndex(test['Date']).month

print(test.head())

#convert categorical to numerical 
categorical_to_numerical(test, 'StateHoliday')
categorical_to_numerical(test, 'SchoolHoliday')

print(test.dtypes)

#Check the data types for each column.
print(test.dtypes)

#now all of our columns are numeric
#get a view on the data
print(test.describe())


#check for NaNs in our selected columns
print("NANs for individual columns")
print("---------------------------")
from collections import Counter
x = {colname : test[colname].isnull().sum() for colname in test.columns}
print(Counter(x).most_common())

#There are 11 missing values in Open column. Let’s have a detailed look at those:
print(test[np.isnan(test['Open'])])


#Do we have any information about store 622? Check train dataset:
print(train[train['Store'] == 622].head())
test[np.isnan(test['Open'])] = 1

#Checking if there are any missing values left.
print("NANs for individual columns")
print("---------------------------")
from collections import Counter
x = {colname : test[colname].isnull().sum() for colname in test.columns}
print(Counter(x).most_common())

#Check for the data types for test dataset.
test.dtypes

#Change the column names of the test dataset to get construct same feature names with the train dataset.
test = test[['Store', 'DayOfWeek', 'Date', 'Year', 'Month', 'Open','Promo', 'StateHoliday', 'SchoolHoliday']]
list(test.columns.values)


#----------------------------------------------------------------------------------------------------------
#Store Dataset

#inspect Store dataset:
print(store.shape)
print(store.dtypes)
print(store.head())
print(store.tail())


#check for duplicates, if exist drop them
print(train.shape, '\n')
train = train.drop_duplicates()
print(train.shape, '\n')
#we see that no rows were dropped, we had no duplicates


print(store.dtypes)
print(store.describe())


#check for missing values
print("NANs for individual columns")
print("---------------------------")
from collections import Counter
x = {colname : store[colname].isnull().sum() for colname in store.columns}
print(Counter(x).most_common())

#promo related values are coorelated. if there is no promo, coorelated values shoul be zeros.
#as we can see from the missing values, that is not correct in our dataset, we should fix it
store.loc[store['Promo2'] == 0, ['Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval']] = 0
store.loc[store['Promo2'] != 0, 'Promo2SinceWeek'] = store['Promo2SinceWeek'].max() - store.loc[store['Promo2'] != 0, 'Promo2SinceWeek']
store.loc[store['Promo2'] != 0, 'Promo2SinceYear'] = store['Promo2SinceYear'].max() - store.loc[store['Promo2'] != 0, 'Promo2SinceYear']

#convert categorical to numerical 
print(store.dtypes)
categorical_to_numerical(store, 'StoreType')
categorical_to_numerical(store, 'Assortment')
store['PromoInterval'].unique()
categorical_to_numerical(store, 'PromoInterval', start_value=0)

print(store.dtypes)
print(store.head())


#last check for missing values
print("NANs for individual columns")
print("---------------------------")
from collections import Counter
x = {colname : store[colname].isnull().sum() for colname in store.columns}
print(Counter(x).most_common())

#handle them with sklearn's imputer
imputer = SimpleImputer().fit(store)
store_imputed = imputer.transform(store)


store_new = pd.DataFrame(store_imputed, columns=store.columns.values)
print("NANs for individual columns")
print("---------------------------")
from collections import Counter
x = {colname : store_new[colname].isnull().sum() for colname in store_new.columns}
print(Counter(x).most_common())

#imputer has done its job, now we have no missing values
#check if "Store" are same in Store and Train datasets
Stores_in_Store = pd.Series(store_new['Store'])
Stores_in_Train = pd.Series(train['Store'])

print(sum(Stores_in_Train.isin(Stores_in_Store) == False))
#they are the same,

#now merge train and store
train_store = pd.merge(train, store_new, how = 'left', on='Store')
train_store.shape
print(train_store.head())
print(train_store.tail())
train_store.isnull().sum()


#merge test and store
test_store = test.reset_index().merge(store_new, how = 'left', on='Store')
test_store.shape
print(test_store.head())
print(test_store.tail())
test_store.isnull().sum()

#now we are ready for modeling, but before we do that we are going to do a visual exploration
'''
# Box plot of 'Sales per Customer'
plt.figure(figsize=(4,3))
train_store['SalesPerCustomer'] = train_store.Sales / train_store.Customers
sns.boxplot(y='SalesPerCustomer', data=train_store)

train_store.drop(columns=['SalesPerCustomer'])

print(train_store.dtypes)
#Box plot of Sales by year
train_store.boxplot(column='Sales', by='Year')
train_store.hist(column='Sales', by='Year')


#Box plot of Sales by month
train_store.boxplot(column='Sales', by='Month')

#sales per day (sundays stores are closed)
train_store.boxplot(column='Sales', by='DayOfWeek')

#sales on holidays
train_store.boxplot(column='Sales', by='StateHoliday')

#sales when schools are closed
train_store.boxplot(column='Sales', by='SchoolHoliday')


# Box plot of 'Customers'
train_store.boxplot(column='Customers', by='Sales')
'''

#LinearRegression fits a linear model with coefficients w = (w1, …, wp) to minimize the residual sum of squares between the observed targets in the dataset, and the targets predicted by the linear approximation.
#A random forest is a meta estimator that fits a number of classifying decision trees on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. The sub-sample size is controlled with the max_samples parameter if bootstrap=True (default), otherwise the whole dataset is used to build each tree.
#GB builds an additive model in a forward stage-wise fashion; it allows for the optimization of arbitrary differentiable loss functions. In each stage a regression tree is fit on the negative gradient of the given loss function.


#models
train_model = train_store.drop(['Customers', 'Date'], axis=1)

print(train_model.head())

test_model = test_store.drop(['Date','Id'], axis=1)


X = train_model.drop('Sales', axis=1)
y = train_model['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = pd.DataFrame(scaler.transform(X_train),columns=X.columns.values)
X_test_scaled = pd.DataFrame(scaler.transform(X_test),columns=X.columns.values)


model_list = {
              'LinearRegression':LinearRegression(),
              'RandomForest_new':RandomForestRegressor(),
              'GradientBoostingRegressor_new':GradientBoostingRegressor()
           }

for  model_name,model in model_list.items():
    model.fit(X_train, y_train)
    model.score(X_test, y_test)
    test_model = pd.DataFrame(test_model)
    submission = {}
    submission = pd.DataFrame()
    submission["Predicted Sales"] = model.predict(test_model)
    submission = submission.reset_index()
    submission.head()
    submission.tail()
    submission.to_csv(model_name, sep=',', index=False)
    submission
