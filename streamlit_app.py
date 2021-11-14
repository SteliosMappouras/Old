import streamlit as st
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import sklearn as sk


#st.title('Data Science Project:')
#st.caption('Stelios Mappouras\nIoannis Volonakis\nSavvina Rousou\nMarios Kyriakides')

#https://www.kaggle.com/xiaoxiao1989/rossmann-sales-prediction-exploration-cleaning

#st.dataframe(store.head())

train = pd.read_csv('train.csv')
store = pd.read_csv('store.csv')

# Dataframe dimensions
print(store.shape)
print(train.shape, '\n')

# Column datatypes
print(store.dtypes,'\n')    
print(train.dtypes)   

# Display first 5 rows of train
print(train.head(), '\n')
print(train.tail(), '\n')

# Display first 5 rows of store
print(store.head(), '\n')
print(store.tail(), '\n')


#wee see that we have some stores that are closed, there is no use for those rows so we can drop them,
#also, we should drop duplicate values if exist

#early data cleaning
#drop duplicates
print(store.shape, '\n')
print(train.shape, '\n')

train = train.drop_duplicates()
store = store.drop_duplicates()

print(store.shape, '\n')
print(train.shape, '\n')

#we see that no rows were dropped, we had no duplicates


# drop stores that are closed in train dataset
train = train[train.Open != 0]
print(train.shape, '\n')

#we can see that we dropped (1017209 - 844392) rows that the stores were closed
print(len(train[train.Customers == 0]))


#sort dataset by store
print(train[train.Customers == 0].sort_values(by=['Store']))


#we see that we have some stores that have 0 sales, we can drop them
len(train[train.Sales == 0])
# after checking the data, decide to drop sales == 0 observations
train = train[train.Sales != 0]

#find average sales per customer
train['SalesPerCustomer'] = train.Sales / train.Customers

print(train)


#study numerical values
#visualize
# Plot histogram grid
train.hist(xrot=-45,figsize=(10,10))
# Clear the text "residue"
##plt.show()

# Plot histogram grid
store.hist(xrot=-45,figsize=(10,10))
# Clear the text "residue"
##plt.show()

# Summarize numerical features
train.describe()
# Promo2Since[Year/Week] - describes the year and calendar week when the store started participating in Promo2
store.describe()


#handle outliers
# Box plot of 'Sales'
plt.figure(figsize=(4,3))
sns.boxplot(y='Sales', data=train)


# Box plot of 'Customers'
plt.figure(figsize=(4,3))
sns.boxplot(y='Customers', data=train)

# Box plot of 'Customers'
plt.figure(figsize=(4,3))
sns.boxplot(y='SalesPerCustomer', data=train)

##plt.show()

#sales per store stats  
train[train.Sales < 1000][['Store','Sales']].describe()
train.groupby('Store')['Sales'].mean().sort_values()


#enixerw giati epiae to 652
train[train.Store == 652]

train[train.Store == 652]['Sales'].describe()
train[train.Store == 652]['Sales'].sort_values()

#reset the indexing
train=train.reset_index()

#function to find outliers
def find_outlier_index(feature):
    main_data = train[['Store',feature]]
    low = find_low_high(feature)["low"]
    high = find_low_high(feature)["high"]
    
    new_low = pd.merge(main_data, low, on='Store', how='left')
    new_low['outlier_low'] = (new_low[feature] < new_low['low'])
    index_low = new_low[new_low['outlier_low'] == True].index
    index_low = list(index_low)
    
    new_high = pd.merge(main_data, high, on='Store', how='left')
    new_high['outlier_high'] = new_high[feature] > new_high['high']
    index_high = new_high[new_high['outlier_high'] == True].index
    index_high = list(index_high)
    
    index_low.extend(index_high)
    index = list(set(index_low))
    return index

def find_low_high(feature):
    # find store specific Q1 - 3*IQ and Q3 + 3*IQ
    IQ = train.groupby('Store')[feature].quantile(0.75)-train.groupby('Store')[feature].quantile(0.25)
    Q1 = train.groupby('Store')[feature].quantile(0.25)
    Q3 = train.groupby('Store')[feature].quantile(0.75)
    low = Q1 - 3*IQ
    high = Q3 + 3*IQ
    low = low.to_frame()
    low = low.reset_index()
    low = low.rename(columns={feature: "low"})
    high = high.to_frame()
    high = high.reset_index()
    high = high.rename(columns={feature: "high"})
    return {'low':low, 'high':high}


#find outliers for sales
print(len(find_outlier_index("Sales")))

#delete sales outliers 
train.drop(find_outlier_index("Sales"), inplace=True, axis=0)

train.shape


#2.3 Box-cox transformation for numerical features and target


from scipy.stats import boxcox
train['Sales'], lam1 = boxcox(train.Sales)
train['Customers'], lam2 = boxcox(train.Customers)
train['AvgPurchasing'], lam3 = boxcox(train.SalesPerCustomer)

print(lam1)
train.Sales.hist(figsize=(4,2))

print(lam2)
train.Customers.hist(figsize=(4,2))

print(lam3)
train.AvgPurchasing.hist(figsize=(4,2))
##plt.show()


#2.4 Missing values of numerical features¶


print(train.select_dtypes(exclude=['object']).isnull().sum(),'\n')
print(store.select_dtypes(exclude=['object']).isnull().sum())

# for competion data, check the 3 missing CompetitionDistance
store[store['CompetitionDistance'].isnull()]

# fill and flag the missing numeric data
store.CompetitionOpenSinceMonth.fillna(0, inplace=True)
store.CompetitionOpenSinceYear.fillna(0, inplace=True)
store.CompetitionDistance.fillna(0, inplace=True)

# flag: indicator variable for missing numeric data
store['CompetitionOpenSinceMonth_missing'] = store.CompetitionOpenSinceMonth.isnull().astype(int)
store['CompetitionOpenSinceYear_missing'] = store.CompetitionOpenSinceYear.isnull().astype(int)
store['CompetitionDistance_missing'] = store.CompetitionDistance.isnull().astype(int)

# check是否当且仅当promo2为0时，Promo2SinceWeek，Promo2SinceYear，Promo2Interval为Nan？
store[store['Promo2']==0][['Promo2SinceWeek','Promo2SinceYear','PromoInterval']].isnull().sum()


# just fill the nan with 0 because it is actually not missing data 
store.Promo2SinceWeek.fillna(0, inplace=True)
store.Promo2SinceYear.fillna(0, inplace=True)
store.PromoInterval.fillna(0, inplace=True)

store.isnull().sum()


#3. Study of categorical features
#3.1 Distribution of categorical features
# Plot bar plot for each categorical feature
plt.figure(figsize=(4,4))
sns.countplot(y='SchoolHoliday', data=train)
plt.figure(figsize=(4,4))
sns.countplot(y='StateHoliday', data=train)

for feature in store.dtypes[store.dtypes=='object'].index:
    plt.figure(figsize=(4,4))
    sns.countplot(y=feature, data=store)


#3.2 Categorical features cleaning

#3.2.1 Structural errors
# Display unique values of 'basement'
train.StateHoliday.unique()
train.StateHoliday.replace(0, '0',inplace=True)


#3.2.2 Missing values
# Display number of missing values by feature (categorical)
print(train.select_dtypes(include=['object']).isnull().sum(), '\n')
print(store.select_dtypes(include=['object']).isnull().sum())

#To do 2: 1) StateHoliday, StoreType, Assortment, needs
#  to be transformed into one-hot-encoding after all the 
# cleaning and feature engineering; 
# 3) CompetitionOpenSinceMonth, etc. 
# may need to transformed to type int
#  in order to match the Year, Month.


#4. Sales, customers, average purchasing segmentated by categorical features

#4.1 Sales on stateholiday are higher, with much more customers but lower avg purchasing
plt.figure(figsize=(4,4))
sns.boxplot(y='Sales', x='StateHoliday', data=train)

plt.figure(figsize=(4,4))
sns.boxplot(y='Customers', x='StateHoliday', data=train)

plt.figure(figsize=(4,4))
sns.boxplot(y='AvgPurchasing', x='StateHoliday', data=train)

#4.2 SchoolHoliday seems have little impact on sales.
#(Note that all schools are closed on public holidays and weekends.)
plt.figure(figsize=(4,4))
sns.boxplot(y='Sales', x='SchoolHoliday', data=train)

#4.3 the transformed sales are usually between 10-14
plt.figure(figsize=(4,4))
sns.boxplot(y='Sales', x='Store', data=train)

#4.4 DoW pattern of sales
plt.figure(figsize=(4,4))
sns.boxplot(y='Sales', x='DayOfWeek', data=train)

#4.5 Promo effect: more sales, more customers, more avg purchasing
plt.figure(figsize=(4,4))
sns.boxplot(y='Sales', x='Promo', data=train)

plt.figure(figsize=(4,4))
sns.boxplot(y='Customers', x='Promo', data=train)

plt.figure(figsize=(4,4))
sns.boxplot(y='AvgPurchasing', x='Promo', data=train)

#4.6 Joining the 2 tables for exploration purpose
train.index = train['Store']
store.index = store['Store']
train = train.drop(['Store'], axis=1)
df_combined = train.join(store)
df_combined = df_combined.reset_index(drop=True)
df_combined.head()

#4.7 Sales v.s. storetype and assortment
# note that the order from the most to the least number in each type: a,d,c,b
sns.boxplot(y='Sales', x='StoreType', data=df_combined)

sns.boxplot(y='Sales', x='Assortment', data=df_combined)

sns.catplot(data=df_combined, x="StoreType", y="Sales", col="Assortment")

# only 9 stores has assortment == 'b'
df_combined[df_combined.Assortment == 'b'].Store.unique()

# only 17 stores has StoreType == 'b'
df_combined[df_combined.StoreType == 'b'].Store.unique()

g = sns.FacetGrid(df_combined, col="StoreType")
g.map(sns.histplot, "Sales")

#5. relationship between numerical features and targets¶

g = sns.FacetGrid(df_combined, col="StoreType")
g.map(plt.scatter, "Customers", "Sales")

sns.lmplot(x='Customers', y='Sales', data=df_combined, hue='StoreType',fit_reg=False)

sns.lmplot(x='AvgPurchasing', y='Sales', data=df_combined, hue='StoreType',fit_reg=False)

sns.lmplot(x='Customers', y='Sales', data=df_combined, hue='Assortment',fit_reg=False)

sns.lmplot(x='AvgPurchasing', y='Sales', data=df_combined, hue='Assortment',fit_reg=False)

sns.lmplot(x='Customers', y='Sales', data=df_combined, hue='Promo',fit_reg=False)

# Calculate correlations between numeric features
correlations = df_combined.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(correlations, dtype=bool)
mask[np.triu_indices_from(mask)] = True

# Make the figsize 10 x 8
plt.figure(figsize=(9,8))
# Plot heatmap of annotated correlations
sns.heatmap(correlations*100, annot=True, fmt='.0f',mask = mask, cbar=False)

#6. Time series exploration

#6.1 Date related feature engineering

def get_date_features(train):
    train['Date'] = pd.to_datetime(train['Date'])
    train['Year'] = train['Date'].dt.year
    train['Month'] = train['Date'].dt.month
    train['Day'] = train['Date'].dt.day
    train['Quarter'] = train['Date'].dt.quarter
    train['Week'] = train['Date'].dt.week
    
    return train

#6.2 Typical store sales study
def get_series(Store_i):
    new_df = df_combined[df_combined.Store == Store_i][['Date','Sales']]
    new_df.index = new_df.Date
    new_df.drop('Date', axis = 1, inplace = True)
    new_series = new_df.T.squeeze()
    return new_series


for i in df_combined.StoreType.unique():
    print(i, df_combined[df_combined.StoreType == i].Store[:1])

new_series_2 = get_series(2)
new_series_85 = get_series(85)
new_series_1 = get_series(1)
new_series_13 = get_series(13)

plt.figure(figsize=(16,2))
new_series_2.plot(style = 'k--')
plt.figure(figsize=(16,2))
new_series_85.plot(style = 'k--')
plt.figure(figsize=(16,2))
new_series_1.plot(style = 'k--')
plt.figure(figsize=(16,2))
new_series_13.plot(style = 'k--')

new_series_2.index = pd.to_datetime(new_series_2.index)
groups = new_series_2.groupby(pd.Grouper(freq='A'))
plt.figure(figsize=(20,3))
a=311
print("Store2 Daily Sales Plot")
for name, group in groups:
    plt.subplot(a) 
    group.plot()
    a+=1
    plt.title(name.year)

groups = new_series_2['2013'].groupby([pd.Grouper(freq="A"),pd.Grouper(freq="Q")])
plt.figure(figsize=(20,3))
a=411
print("Store2 Daily Sales Plot")
for name, group in groups:
    plt.subplot(a) 
    group.plot()
    a+=1
    plt.title(name)

groups = new_series_2['2014'].groupby([pd.Grouper(freq="A"),pd.Grouper(freq="Q")])
plt.figure(figsize=(20,3))
a=411
for name, group in groups:
    plt.subplot(a) 
    group.plot()
    a+=1
    plt.title(name)
    
groups = new_series_2['2015'].groupby([pd.Grouper(freq="A"),pd.Grouper(freq="Q")])
plt.figure(figsize=(20,3))
a=411
for name, group in groups:
    plt.subplot(a) 
    group.plot()
    a+=1
    plt.title(name)

groups = new_series_2['2013'].groupby([pd.Grouper(freq="A"),pd.Grouper(freq="M")])
plt.figure(figsize=(15,6))
a=611
print("Store2 Daily Sales Plot")
i = 1
for name, group in groups:
    if i>6:
        break
    else:
        plt.subplot(a) 
        group.plot()
        a+=1
        i+=1
        plt.title(name)

plt.figure(figsize=(15,6))
i = 1
a=611
for name, group in groups:
    if i<=6:
        i+=1
    else:
        plt.subplot(a) 
        group.plot()
        a+=1
        i+=1
        plt.title(name)

#sales time series basically has a weekly seasonality, with typical pattern around DoW, MoY
#6.3 All store sales time series study

df_combined.Date = pd.to_datetime(df_combined.Date)

daily_sales_sum = df_combined.groupby(['Date'])['Sales'].sum()
daily_sales_mean = df_combined.groupby(['Date'])['Sales'].mean()
daily_sales_median = df_combined.groupby(['Date'])['Sales'].median()
daily_sales_max = df_combined.groupby(['Date'])['Sales'].max()
daily_sales_min = df_combined.groupby(['Date'])['Sales'].min()

print("All stores total monthly sales - by Year")
plt.figure(figsize=(16,2))
daily_sales_sum['2013'].groupby([pd.Grouper(freq="A"),pd.Grouper(freq="M")]).sum().plot()

plt.figure(figsize=(16,2))
daily_sales_sum["2014"].groupby([pd.Grouper(freq="A"),pd.Grouper(freq="M")]).sum().plot()

plt.figure(figsize=(16,2))
daily_sales_sum["2015"].groupby([pd.Grouper(freq="A"),pd.Grouper(freq="M")]).sum().plot()

groups = daily_sales_sum["2013"].groupby([pd.Grouper(freq="A"),pd.Grouper(freq="M")])

plt.figure(figsize=(15,6))
a=611
print("2013 All Store Daily Total Sales Plot - by Month")
i = 1
for name, group in groups:
    if i>6:
        break
    else:
        plt.subplot(a) 
        group.plot()
        a+=1
        i+=1
        plt.title(name)

plt.figure(figsize=(15,6))
i = 1
a=611
for name, group in groups:
    if i<=6:
        i+=1
    else:
        plt.subplot(a) 
        group.plot()
        a+=1
        i+=1
        plt.title(name)
        

groups = daily_sales_mean["2014"].groupby([pd.Grouper(freq="A"),pd.Grouper(freq="M")])

plt.figure(figsize=(15,6))
a=611
print("2014 All Store Daily Total Sales Plot - by Month")
i = 1
for name, group in groups:
    if i>6:
        break
    else:
        plt.subplot(a) 
        group.plot()
        a+=1
        i+=1
        plt.title(name)

plt.figure(figsize=(15,6))
i = 1
a=611
for name, group in groups:
    if i<=6:
        i+=1
    else:
        plt.subplot(a) 
        group.plot()
        a+=1
        i+=1
        plt.title(name)

groups = daily_sales_mean["2015"].groupby([pd.Grouper(freq="A"),pd.Grouper(freq="M")])

plt.figure(figsize=(15,6))
a=611
print("2015 All Store Daily Total Sales Plot - by Month")
i = 1
for name, group in groups:
    if i>6:
        break
    else:
        plt.subplot(a) 
        group.plot()
        a+=1
        i+=1
        plt.title(name)

plt.figure(figsize=(15,6))
i = 1
a=611
for name, group in groups:
    if i<=6:
        i+=1
    else:
        plt.subplot(a) 
        group.plot()
        a+=1
        i+=1
        plt.title(name)

#6.4 Statistics over window period
'''
groups = df_combined.groupby(['Year','Month'])['Sales'].mean()
plt.figure(figsize=(10,3))
a = plt.subplot(1,1,1)
#plt.subplot(131) 
#plt.title('Monthly mean plot',color='blue') 
line1=groups.plot(label = 'mean')

groups = df_combined.groupby(['Year','Month'])['Sales'].median()
line2=groups.plot(label = 'median')

groups = df_combined.groupby(['Year','Month'])['Sales'].max()
line3=groups.plot(label = 'max')

groups = df_combined.groupby(['Year','Month'])['Sales'].min()
line4=groups.plot(label = 'min')

handles, labels = a.get_legend_handles_labels()
a.legend(handles[::-1], labels[::-1])
plt.title("overall sales: monthly statistics",color='blue')

groups = df_combined.groupby(['Year'])['Sales'].mean()
plt.figure(figsize=(10,3))
a = plt.subplot(1,1,1)
#plt.subplot(131) 
#plt.title('Monthly mean plot',color='blue') 
line1=groups.plot(label = 'mean')

groups = df_combined.groupby(['Year'])['Sales'].median()
line2=groups.plot(label = 'median')

groups = df_combined.groupby(['Year'])['Sales'].max()
line3=groups.plot(label = 'max')

groups = df_combined.groupby(['Year'])['Sales'].min()
line4=groups.plot(label = 'min')

handles, labels = a.get_legend_handles_labels()
a.legend(handles[::-1], labels[::-1])
plt.title("overall sales: yearly statistics",color='blue')

groups = new_series_2.groupby(pd.Grouper(freq="W")).mean()
plt.figure(figsize=(10,3))
a = plt.subplot(1,1,1)
#plt.subplot(131) 
#plt.title('Monthly mean plot',color='blue') 
line1=groups.plot(label = 'mean')

groups = new_series_2.groupby(pd.Grouper(freq="W")).median()
line2=groups.plot(label = 'median')

groups = new_series_2.groupby(pd.Grouper(freq="W")).max()
line3=groups.plot(label = 'max')

groups = new_series_2.groupby(pd.Grouper(freq="W")).min()
line4=groups.plot(label = 'min')

handles, labels = a.get_legend_handles_labels()
a.legend(handles[::-1], labels[::-1])


groups = new_series_2.groupby(pd.Grouper(freq="M")).mean()
plt.figure(figsize=(10,3))
a = plt.subplot(1,1,1)
#plt.subplot(131) 
#plt.title('Monthly mean plot',color='blue') 
line1=groups.plot(label = 'mean')

groups = new_series_2.groupby(pd.Grouper(freq="M")).median()
line2=groups.plot(label = 'median')

groups = new_series_2.groupby(pd.Grouper(freq="M")).max()
line3=groups.plot(label = 'max')

groups = new_series_2.groupby(pd.Grouper(freq="M")).min()
line4=groups.plot(label = 'min')

handles, labels = a.get_legend_handles_labels()
a.legend(handles[::-1], labels[::-1])


groups = new_series_2.groupby(pd.Grouper(freq="Q")).mean()
plt.figure(figsize=(10,3))
a = plt.subplot(1,1,1)
#plt.subplot(131) 
#plt.title('Monthly mean plot',color='blue') 
line1=groups.plot(label = 'mean')

groups = new_series_2.groupby(pd.Grouper(freq="Q")).median()
line2=groups.plot(label = 'median')

groups = new_series_2.groupby(pd.Grouper(freq="Q")).max()
line3=groups.plot(label = 'max')

groups = new_series_2.groupby(pd.Grouper(freq="Q")).min()
line4=groups.plot(label = 'min')

handles, labels = a.get_legend_handles_labels()
a.legend(handles[::-1], labels[::-1])
'''

#6.5 Time series lag plot
def lag_n_plot(series, n):
    series_lag_n = series.shift(n)
    df_from_series = pd.DataFrame(series)
    df_from_series = df_from_series.rename(columns={'Sales':'Sales_t'})
    df_from_series_lag_n = pd.DataFrame(series_lag_n)
    df_from_series_lag_n = df_from_series_lag_n.rename(columns={'Sales':'Sales_t-n'})
    new_df = pd.concat([df_from_series, df_from_series_lag_n], axis=1)
    plt.title('Lag %d plot' %(n))
    #plt.figure(figsize=(3,3))
    plt.scatter(y = "Sales_t", x = "Sales_t-n", data=new_df, alpha = 0.5)

#lag plot of All Store daily sales sum
print('lag plot of All Store daily sales sum')
plt.figure(figsize=(16,2))
plt.subplot(151) 
lag_n_plot(daily_sales_sum, 1)

plt.subplot(152) 
lag_n_plot(daily_sales_sum, 7)

plt.subplot(153) 
lag_n_plot(daily_sales_sum, 14)

plt.subplot(154) 
lag_n_plot(daily_sales_sum, 28)

plt.subplot(155) 
lag_n_plot(daily_sales_sum, 90)



#lag plot of All Store daily sales mean
print('lag plot of All Store daily sales mean')
plt.figure(figsize=(16,2))
plt.subplot(151) 
lag_n_plot(daily_sales_mean, 1)

plt.subplot(152) 
lag_n_plot(daily_sales_mean, 7)

plt.subplot(153) 
lag_n_plot(daily_sales_mean, 14)

plt.subplot(154) 
lag_n_plot(daily_sales_mean, 28)

plt.subplot(155) 
lag_n_plot(daily_sales_mean, 90)

#lag plot of Store2 daily sales
print('lag plot of Store2 daily sales')
plt.figure(figsize=(16,2))
plt.subplot(151) 
lag_n_plot(new_series_2, 1)

plt.subplot(152) 
lag_n_plot(new_series_2, 7)

plt.subplot(153) 
lag_n_plot(new_series_2, 14)

plt.subplot(154) 
lag_n_plot(new_series_2, 28)

plt.subplot(155) 
lag_n_plot(new_series_2, 90)



#6.6 Time series autocorrelation plot
from pandas.plotting import autocorrelation_plot
print('autocorrelation plot of Store2 daily sales')
plt.figure(figsize=(20,4))
plt.xticks([x for x in range(900) if x % 28 == 0]) 
autocorrelation_plot(new_series_2)

print('autocorrelation plot of All Store daily sales mean')
plt.figure(figsize=(20,4))
plt.xticks([x for x in range(900) if x % 28 == 0])  
autocorrelation_plot(daily_sales_mean)

print('autocorrelation plot of All Store daily sales sum')
plt.figure(figsize=(20,4))
plt.xticks([x for x in range(900) if x % 28 == 0]) 
autocorrelation_plot(daily_sales_sum)

##plt.show()

#7. Save the table¶
df_combined.to_csv('df_combined_cleaned.csv', index=None)
    
