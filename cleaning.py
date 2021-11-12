import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import sklearn as sk
from sklearn import model_selection

import xgboost as xgb



from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

# Import datast 
store = pd.read_csv('store.csv')
train = pd.read_csv('train.csv', index_col='Date', parse_dates=True)
test = pd.read_csv('test.csv')
train.shape, test.shape, store.shape

print("store shape: ", store.shape, "\n")
print("train shape: ", train.shape, "\n")
print("test shape: ", test.shape, "\n")

print("store head: ", store.head(), "\n")
print("train head: ", train.head(), "\n")
print("test head: ", test.head(), "\n")


#1.1: Trends & Seasonility
#How the sales vary with month, promo(First promotional Offer), promo2(Second Promotional Offer) and years.
# Extract Year, Month, Day, Wee columns 
train['Year'] = train.index.year
train['Month'] = train.index.month
train['Day'] = train.index.day
train['WeekofYear'] = pd.Int64Index(train.index.isocalendar().week)

train['SalesPerCustomer'] = train['Sales']/train['Customers']

# Checking the data when the store is closed 
train_store_closed = train[(train.Open == 0)]
train_store_closed.head()

# Check when the store was closed 
train_store_closed.hist('DayOfWeek')

#From this chart, we could see that, 7th day store was mostly clodes. It is Sunday and makes sense.

# Check whether there school was closed for holyday 
train_store_closed['SchoolHoliday'].value_counts().plot(kind='bar')


#Here 1 is school closed day and it pretty low. And 0 is None.

# Check whether there school was closed for holyday 
train_store_closed['StateHoliday'].value_counts().plot(kind='bar')
#Here, The state is closed for (a= Public holyday, b = Easter holyday, c = Christmas and 0 is None)

# Check the null values
# In here there is no null value 
train.isnull().sum()

# Number of days with closed stores
train[(train.Open == 0)].shape[0]

# Okay now check No. of dayes store open but sales zero ( It might be caused by external refurbishmnent)
train[(train.Open == 1) & (train.Sales == 0)].shape[0]

#------------------------------------------------------------------------------

# Work with store data 
store.head()

# Check null values 
# Most of the columns has null values 
store.isnull().sum()

# Replacing missing values for Competiton distance with median
store['CompetitionDistance'].fillna(store['CompetitionDistance'].median(), inplace=True)

# No info about other columns - so replcae by 0
store.fillna(0, inplace=True)

# Again check it and now its okay 
store.isnull().sum().sum()


#-------------------------------------------------

# Work with test data 
test.head()

# check null values ( Only one feature Open is empty)
test.isnull().sum()

# Assuming stores open in test
test.fillna(1, inplace=True)

# Again check 
test.isnull().sum().sum()

# Join train and store table 
train_store_joined = pd.merge(train, store, on='Store', how='inner')
train_store_joined.head()

train_store_joined.groupby('StoreType')['Customers', 'Sales', 'SalesPerCustomer'].sum().sort_values('Sales', ascending='desc')

# Closed and zero-sales observations 
train_store_joined[(train_store_joined.Open == 0) | (train_store_joined.Sales==0)].shape

#So, we have 172,871 observations when the stores were closed or have zero sales.

# Open & Sales >0 stores
train_store_joined_open = train_store_joined[~((train_store_joined.Open ==0) | (train_store_joined.Sales==0))]
train_store_joined_open


#Correlation Analysis
plt.figure(figsize=(20, 10))
sns.heatmap(train_store_joined.corr(), annot=True)

#From the above chart we can see a strong positive correlation between the amount of Sales and Customers visiting the store. We can also observe a positive correlation between a running promotion (Promo = 1) and number of customers.¶

# Now plot the sales trend over the month 
sns.factorplot(data = train_store_joined_open, x='Month', y='Sales',
              col ='Promo', hue='Promo2', row='Year')

# Sales and trend over days
sns.factorplot(data= train_store_joined_open, x='DayOfWeek', y="Sales",
              hue='Promo')

#From the above chart, 0 represents sales and 1 represents promotin in a week. Promotions are not given in weekend (Saturday and Sunday). Because peoples are goinf to buy their household things on the weekend and wothout promotion sales increased in a dramatic way. Promotion are highest on monday and as well as sales are high on that day

#Insights
#1. Storetype a has highest customer and sales
#2. Storetype b has highest SalesPerCustomer
#3. There is no promotion offer in Saturday and Sunday
#4. Customers are going to buy their goods in tuesday on promotional offer.

#2. Time Series Analysis

#In this section we will consider only one store from each store type(a, b, c, d).

pd.plotting.register_matplotlib_converters()

#Register pandas formatters and converters with matplotlib.

#This function modifies the global matplotlib.units.registry dictionary. pandas adds custom converters for
#pd.Timestamp
#pd.Period
#np.datetime64
#datetime.datetime
#datetime.date
#datetime.time


# Data Preparation: input should be float type 

# our Sales data is int type so lets make it float
train['Sales'] = train['Sales'] * 1.00
train['Sales'].head()

train.Store.unique()

# Assigning one store from each category
sales_a = train[train.Store == 2]['Sales']
sales_b = train[train.Store == 85]['Sales'].sort_index(ascending = True) 
sales_c = train[train.Store == 1]['Sales']
sales_d = train[train.Store == 13]['Sales']

frame, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize = (20, 16))

# Visualize Trend 
sales_a.resample('w').sum().plot(ax = ax1)
sales_b.resample('w').sum().plot(ax = ax2)
sales_c.resample('w').sum().plot(ax = ax3)
sales_d.resample('w').sum().plot(ax = ax4)


# will be used to resample the speed column of our DataFrame
#The 'W' indicates we want to resample by week. At the bottom of this post is a summary of different time frames.
# You could use for Day = d, MOnth = m, Year = y


#From the above chart we could see sales of store type A, C has highest sales at the end of the year. December months has christmas season. So, that they get highes salary. At the end of the month their sell decrease. We can not find semiler trend for store B and D, it could be there is no data for that time perion. Possible reason is "store closed".


#stationarity of Time Seriese
#Stationarity means that the statistical properties of a time series do not change over time. Some stationary data is (constant mean, constant variance and constant covariance with time).

#There are 2 ways to test the stationarity of time series
#A) Rolling Mean: Visualization
#B) Dicky - Fuller test: Statistical test
#A) Rolling Mean: A rolling analysis of a time series model is often used to assess the model's stability over time. The window is rolled (slid across the data) on a weekly basis, in which the average is taken on a weekly basis. Rolling Statistics is a visualization test, where we can compare the original data with the rolled data and check if the data is stationary or not.
#B) Dicky -Fuller test: This test provides us the statistical data such as p-value to understand whether we can reject the null hypothesis. If p-value is less than the critical value (say 0.5), we will reject the null hypothesis and say that data is stationary.

# lets create a functions to test the stationarity 
def test_stationarity(timeseries):
    # Determine rolling statestics 
    roll_mean = timeseries.rolling(window=7).mean()
    roll_std = timeseries.rolling(window=7).std()
    
    # plotting rolling statestics 
    plt.subplots(figsize = (16, 6))
    orginal = plt.plot(timeseries.resample('w').mean(), color='blue',linewidth= 3, label='Orginal')
    roll_mean = plt.plot(roll_mean.resample('w').mean(), color='red',linewidth= 3, label='Rolling Mean')
    roll_mean = plt.plot(roll_std.resample('w').mean(), color='green',linewidth= 3, label='Rolling Std')
    
    plt.legend(loc='best')
    
    # Performing Dickey-Fuller test 
    print('Result of Dickey-Fuller test:')
    result= adfuller(timeseries, autolag='AIC')
    
    print('ADF Statestics: %f' %result[0])
    print('P-value: %f' %result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(key, value)

test_stationarity(sales_a)
test_stationarity(sales_b)
test_stationarity(sales_c)
test_stationarity(sales_d)

#from above charts we could observe that, mean and variance of the data are not change most over time. So, we do not compute any transformation.

#Lets create trends and seasonality

# plotting trends and seasonality 

def plot_timeseries(sales,StoreType):

    fig, axes = plt.subplots(2, 1, sharex=True, sharey=False)
    fig.set_figheight(6)
    fig.set_figwidth(20)

    decomposition= seasonal_decompose(sales, model = 'additive',freq=365)

    estimated_trend = decomposition.trend
    estimated_seasonal = decomposition.seasonal
    estimated_residual = decomposition.resid
    
    axes[1].plot(estimated_seasonal, 'g', label='Seasonality')
    axes[1].legend(loc='upper left');
    
    axes[0].plot(estimated_trend, label='Trend')
    axes[0].legend(loc='upper left');

    plt.title('Decomposition Plots')

plot_timeseries(sales_a, 'a')
plot_timeseries(sales_b, 'b')
plot_timeseries(sales_c, 'c')
plot_timeseries(sales_d, 'd')

#From the above plots, we can see that there is seasonality and trend present in our data. So, we'll use forecasting models that take both of these factors into consideration. For example, SARIMAX and Prophet.

#Time Series Forcusting¶
#Evaluation Matrics
#1. MAE - Mean Absolute Error: It is the average of the absolute difference between the predicted values and observed values.
#2. RMSE - Root Mean Square Error: It is the square root of the average of squared differences between the predicted values and observed values.

















#MOdel 2: XGBoost
#Now we will drop columns that are correlated (e.g Customers, SalePerCustomer) in addition to merging similar columns into one column (CompetitionOpenSinceMonth, CompetitionOpenSinceYear).

# Dropping Customers and Sale per customer
ts_xgboost = train_store_joined.copy()
ts_xgboost = ts_xgboost.drop(['Customers', 'SalesPerCustomer', 'PromoInterval'], axis=1)

ts_xgboost.head()
# Here we do not have any categorical variables so we do not have to convert them into numerical to use in XGBoost

# Combining similar columns into one column and dropping old columns
ts_xgboost['CompetitionOpen'] = 12 * (ts_xgboost.Year - ts_xgboost.CompetitionOpenSinceYear) + (ts_xgboost.Month - ts_xgboost.CompetitionOpenSinceMonth)
ts_xgboost['PromoOpen'] = 12 * (ts_xgboost.Year - ts_xgboost.Promo2SinceYear) + (ts_xgboost.WeekofYear - ts_xgboost.Promo2SinceWeek) / 4.0
ts_xgboost = ts_xgboost.drop(["CompetitionOpenSinceMonth", "CompetitionOpenSinceYear"], axis = 1)
ts_xgboost = ts_xgboost.drop(["Promo2SinceWeek", "Promo2SinceYear"], axis = 1)

# Converting categorical cols to numerical cols and removing old cols
mappings = {0:0, "0": 0, "a": 1, "b": 1, "c": 1}
ts_xgboost["StateHoliday_cat"] = ts_xgboost["StateHoliday"].map(mappings)
ts_xgboost["StoreType_cat"] = ts_xgboost["StoreType"].map(mappings)
ts_xgboost["Assortment_cat"] = ts_xgboost["Assortment"].map(mappings)
ts_xgboost = ts_xgboost.drop(["StateHoliday", "StoreType", "Assortment"], axis = 1)

# Splitting the data
features = ts_xgboost.drop(["Sales"], axis = 1)
target = ts_xgboost["Sales"]

ts_xgboost.to_csv('cleaned_dataset_new.csv', index=None)



X_train, X_test, y_train, y_test = model_selection.train_test_split(features, target, test_size = 0.20)

#Baseline XGBoost

# Tuning parameters - using default metrics
params = {'max_depth':6, "booster": "gbtree", 'eta':0.3, 'objective':'reg:linear'} 

dtrain = xgb.DMatrix(X_train, y_train)
dtest = xgb.DMatrix(X_test, y_test)
watchlist = [(dtrain, 'train'), (dtest, 'eval')]

# Training the model
xgboost = xgb.train(params, dtrain, 100, evals=watchlist,early_stopping_rounds= 100, verbose_eval=True)
         
# Making predictions
preds = xgboost.predict(dtest)

# RMSE of model
rms_xgboost = sqrt(mean_squared_error(y_test, preds))
print("Root Mean Squared Error for XGBoost:", rms_xgboost)

#Hypertuning XGBoost
#Now let's try to decrease the RMSE of XGBoost by passing different values for our hyperparameters in the XGBoost model.
#eta: It defines the learning rate i.e step size to learn the data in the gradient descent modeling (the basis for XGBoost). The default value is 0.3 but we want to keep the learning rate low to avoid overfitting. So, we'll choose 0.2 as eta
#max_depth: Maximum depth of a tree. The default value is 6 but we want our model to be more complex and find good predictions. So, let's choose 10 as max depth.
#gamma: Minimum loss reduction required to make a further partition on a leaf node of the tree. The larger gamma is, the more conservative the algorithm will be. The default value is 0, let's choose a little higher value so as to get good predictions

# Tuning parameters
params_2 = {'max_depth':10, 'eta':0.1,  'gamma': 2}

dtrain = xgb.DMatrix(X_train, y_train)
dtest = xgb.DMatrix(X_test, y_test)
watchlist = [(dtrain, 'train'), (dtest, 'eval')]

# Training the model
xgboost_2 = xgb.train(params_2, dtrain, 100, evals=watchlist,early_stopping_rounds= 100, verbose_eval=True)
         
# Making predictions
preds_2 = xgboost_2.predict(dtest)

# RMSE of model
rms_xgboost_2 = sqrt(mean_squared_error(y_test, preds_2))
print("Root Mean Squared Error for XGBoost:", rms_xgboost_2)

# Let's see the feature importance
fig, ax = plt.subplots(figsize=(10,10))
xgb.plot_importance(xgboost_2, max_num_features=50, height=0.8, ax=ax)

#Final XGBoost Model:
#After hypertuning, we were able to reduce RMSE from 1223.31 to 1176.20 which is great! Now, let's compare the performance of all models

#Results
# Comparing performance of above three models - through RMSE
rms_arima = format(float(rms_arima))
rms_xgboost_2 = format(float(rms_xgboost_2))

model_errors = pd.DataFrame({
    "Model": ["SARIMA",  "XGBoost"],
    "RMSE": [rms_arima, rms_xgboost_2]
})

model_errors.sort_values(by = "RMSE")

print(model_errors)


