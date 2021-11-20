import streamlit as st

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
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score



def categorical_to_numerical(df, colname, start_value=0):
    while df[colname].dtype == object:
        myval = start_value # factor starts at "start_value".
        for sval in df[colname].unique():
            df.loc[df[colname] == sval, colname] = myval
            myval += 1
        df[colname] = df[colname].astype(int, copy=False)
    print('levels :', df[colname].unique(), '; data type :', df[colname].dtype)


st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)

st.markdown("""
<nav class="navbar fixed-top navbar-expand-lg navbar-dark" style="background-color: #8A6F6F;">
  <a class="navbar-brand" target="_blank"><strong>Data Science Project</strong></a>
  <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
    <span class="navbar-toggler-icon"></span>
  </button>
  <div class="collapse navbar-collapse" id="navbarNav">
    <ul class="navbar-nav">
      <li class="nav-item active">
        <a class="nav-link disabled" href="#">Home <span class="sr-only">(current)</span></a>
      </li>
      <li class="nav-item">
        <a class="nav-link" href="https://docs.google.com/document/d/17O8pSYi2p1Odq_7A-E2YFDlXCnAHfYIe/edit" target="_blank">Problem</a>
      </li>
    </ul>
  </div>
</nav>
""", unsafe_allow_html=True)

#variables
train = pd.read_csv("train.csv", sep=',', parse_dates=['Date'],
                    dtype={'StateHoliday': str, 'SchoolHoliday':str})
test = pd.read_csv("test.csv", sep=",", index_col = 'Id', parse_dates=['Date'],
                            dtype={'StateHoliday': str, 'SchoolHoliday':str})
store = pd.read_csv("store.csv", sep=",", dtype={'StoreType': str,'Assortment': str,'PromoInterval': str})

st.title('''**Πρόβλεψη μελλοντικών πωλήσεων**''')

st.session_state.workflow = st.sidebar.selectbox('Select a Data Science Life Cycle', ['Business problem', 'Data acquisition', 'Data preparation', 'Exploratory Data analysis', 'Data modeling', 'Visualization & Communication', 'Deployment & Maintenance'] )

if st.session_state.workflow == 'Business problem':
        '''
        st.session_state.data_type=st.header('**Step 1 - Business Problem**') 
        st.session_state.data_type=st.subheader('Πρόβλημα') 
        st.session_state.data_type=st.caption("""Χρησιμοποιώντας τα διαθέσιμα δεδομένα 
        (ιστορικά δεδομένα πωλήσεων) να ακολουθηθούν τα αναγκαία βήματα και 
        αναπτυχθεί ένα μοντέλο όπου θα μπορεί να προβλέψει τις πωλήσεις που 
        μπορούν να γίνουν μελλοντικά (στο σχετικό test dataset).""")

        st.session_state.data_type=st.subheader('Δεδομένα') 
        st.session_state.data_type=st.caption("""Ηistorical data including Sales, Historical data excluding Sales and Supplemental information about the stores """)
            '''












if st.session_state.workflow == 'Data acquisition':
        st.session_state.data_type=st.header('**Step 2 - Data acquisition**')
        st.session_state.data_type=st.subheader('Collection of Datasets')
        st.session_state.data_type=st.write('File **train.csv** - Historical data including Sales - Sample')
        train = pd.read_csv("train.csv", sep=',', parse_dates=['Date'],
                    dtype={'StateHoliday': str, 'SchoolHoliday':str})

        sample = train.head(200)
        st.write(sample)

        st.session_state.data_type=st.write('File **test.csv** - Historical data excluding Sales - Sample')
        test = pd.read_csv("test.csv", sep=",", index_col = 'Id', parse_dates=['Date'],
                            dtype={'StateHoliday': str, 'SchoolHoliday':str})
        sample = test.head(200)
        st.write(sample)

        st.session_state.data_type=st.write('File **store.csv** - Supplemental information about the stores')
        store = pd.read_csv("store.csv", sep=",", dtype={'StoreType': str,'Assortment': str,'PromoInterval': str})
        st.write(store)


if st.session_state.workflow == 'Data preparation':
        st.session_state.data_type=st.header('**Step 3 - Data preparation**')
        st.session_state.data_type=st.subheader('Cleaning')
        st.session_state.data_type=st.write('Inconsistent data types, misspelled attributes, missing values, duplicated values')

        st.session_state.data_type=st.subheader('Transformation')
        st.session_state.data_type=st.write('Normalization, aggregation')

        st.session_state.data_type=st.subheader('Reduction')
        st.session_state.data_type=st.write('Reduction in size')


        st.session_state.data_type=st.write('We already started fixing inconsistent data types while importing the datasets, we parsed the data dypes as string in order to be able to handle it easier later.')
        st.session_state.data_type=st.write('As we saw from visualizing some rows of each dataset, we have some categorical data, which we can turn into numerical: The date can break into Year/Month, we already have day of week in the dataset. Also, for example assortment is alphabetical,\'a\' -> 0, \'b\' -> 1, etc.')
        st.session_state.data_ypes=st.write('This function is created to turn categorical column into numerical:\n def categorical_to_numerical(df, colname, start_value=0):\
                                 while df[colname].dtype == object:\
        myval = start_value # factor starts at "start_value".\
        for sval in df[colname].unique():\
            df.loc[df[colname] == sval, colname] = myval\
            myval += 1\
        df[colname] = df[colname].astype(int, copy=False)')


        st.session_state.data_type=st.subheader('Train Dataset:')
        st.session_state.data_types=st.write('Check for duplicates and drop them')
        st.session_state.data_types=st.write("Train shape before Dropping:",train.shape)
        train = train.drop_duplicates()
        st.session_state.data_types=st.write("After Dropping:",train.shape)
        st.session_state.data_types=st.write("As we can see, no rows were dropped, we had no duplicates in train dataset.")


        st.session_state.data_types=st.write("Open Stores:",sum(train['Open'] == 1))
        st.session_state.data_types=st.write("Closed Stores:",sum(train['Open'] == 0))
        st.session_state.data_types=st.write("drop stores that are closed because they are useless for our forecast as they have no sales")
        train = train[train.Open != 0]
        st.session_state.data_types=st.write("Open Stores:",sum(train['Open'] == 1))
        st.session_state.data_types=st.write("Closed Stores:",sum(train['Open'] == 0))

        st.session_state.data_types=st.write("Next, we want to break the Date column into Year and Month, then drop date.")
        train['Year'] = pd.DatetimeIndex(train['Date']).year
        train['Month'] = pd.DatetimeIndex(train['Date']).month

        sample = train.head(200)
        st.write(sample)

        st.write('Now we want to convert remaining categorical data into numerical: ', train.dtypes.astype(str))
        st.write('We use our function: categorical_to_numerical, for StateHoliday, SchoolHoliday')
        categorical_to_numerical(train,'StateHoliday')
        categorical_to_numerical(train,'SchoolHoliday')
        st.write('Train Data Types After:', train.dtypes.astype(str))

        st.write('check for empty (NaN values) for each column')
        st.write(train.isnull().sum())

        st.write('all values are zero, we have no empty values.')



        st.write('specify the columns that are going to be used on the model.')


        #we have no NANs in train dataset
        #select columns for the training data
        train = train[['Store', 'DayOfWeek', 'Date', 'Year', 'Month', 'Customers', 'Open','Promo', 'StateHoliday', 'SchoolHoliday', 'Sales']]
        st.write(list(train.columns.values))

        st.write("Results of Train:","Stats:",train.describe(), "First Rows:",train.head(),"Last Rows:", train.tail())













        st.session_state.data_type=st.subheader('Test Dataset:')
        st.session_state.data_types=st.write('Repeat the same preparation process')

        st.session_state.data_types=st.write('Check for duplicates and drop them')
        st.session_state.data_types=st.write("Test shape before Dropping:",test.shape)
        test = test.drop_duplicates()
        st.session_state.data_types=st.write("After Dropping:",test.shape)
        st.session_state.data_types=st.write("As we can see, no rows were dropped, we had no duplicates in test dataset.")


        st.session_state.data_types=st.write("Open Stores:",sum(test['Open'] == 1))
        st.session_state.data_types=st.write("Closed Stores:",sum(test['Open'] == 0))
        st.session_state.data_types=st.write("drop stores that are closed because they are useless for our forecast as they have no sales")
        test = test[test.Open != 0]
        st.session_state.data_types=st.write("Open Stores:",sum(test['Open'] == 1))
        st.session_state.data_types=st.write("Closed Stores:",sum(test['Open'] == 0))

        st.session_state.data_types=st.write("Next, we want to break the Date column into Year and Month, then drop date.")
        test['Year'] = pd.DatetimeIndex(test['Date']).year
        test['Month'] = pd.DatetimeIndex(test['Date']).month
        test.drop(columns=['Date'])
        sample = test.head(200)
        st.write(sample)

        st.write('Now we want to convert remaining categorical data into numerical: ', test.dtypes.astype(str))
        st.write('We use our function: categorical_to_numerical, for StateHoliday, SchoolHoliday')
        categorical_to_numerical(test,'StateHoliday')
        categorical_to_numerical(test,'SchoolHoliday')
        categorical_to_numerical(test,'Open')

        st.write('Test Data Types After:', test.dtypes.astype(str))

        st.write('check for empty (NaN values) for each column')
        st.write(test.isnull().sum())

     
        st.write('There are 11 missing values in Open column, lets see from which store they come:')
        st.write(test[np.isnan(test['Open'])])
        st.write('It is from the store 622, lets search if there are is any info about this store in train dataset:')
        st.write(train[train['Store'] == 622].head())
        st.write('We found info, so we will assume that the store is open.')
        test[np.isnan(test['Open'])] = 1

        st.write("Check if there are still missing values")
        st.write(test.isnull().sum())


        st.write('Test Data Types:', test.dtypes.astype(str))


        st.write('specify the columns that are going to be used on the model and change the order to match train dataset.')
        #select columns for the testing data
        test = test[['Store', 'DayOfWeek', 'Date', 'Year', 'Month', 'Open','Promo', 'StateHoliday', 'SchoolHoliday']]
        st.write(list(test.columns.values))

        st.write("Results of Test:")
        st.write("Stats:",test.describe(), "First Rows:",test.head(),"Last Rows:", test.tail())












        
        st.session_state.data_type=st.subheader('Store Dataset:')
        st.session_state.data_types=st.write('Repeat the same preparation process')

        st.session_state.data_types=st.write('Check for duplicates and drop them')
        st.session_state.data_types=st.write("Store shape before Dropping:",store.shape)
        store = store.drop_duplicates()
        st.session_state.data_types=st.write("After Dropping:",store.shape)
        st.session_state.data_types=st.write("As we can see, no rows were dropped, we had no duplicates in store dataset.")

        st.write('check for empty (NaN values) for each column')
        st.write(store.isnull().sum())



        st.write(store.describe())

        st.write("Promo related values are coorelated. if there is no promo, coorelated values shoul be zeros.")
        st.write("As we can see from the missing values, that is not correct in our dataset, we should fix it")
        store.loc[store['Promo2'] == 0, ['Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval']] = 0
        store.loc[store['Promo2'] != 0, 'Promo2SinceWeek'] = store['Promo2SinceWeek'].max() - store.loc[store['Promo2'] != 0, 'Promo2SinceWeek']
        store.loc[store['Promo2'] != 0, 'Promo2SinceYear'] = store['Promo2SinceYear'].max() - store.loc[store['Promo2'] != 0, 'Promo2SinceYear']

        st.write(store.describe())

        st.write('Now we want to convert remaining categorical data into numerical: ', store.dtypes.astype(str))
        st.write('We use our function: categorical_to_numerical, for StoreType, assortment, Promo Interval')
        categorical_to_numerical(store, 'StoreType')
        categorical_to_numerical(store, 'Assortment')
        store['PromoInterval'].unique()
        categorical_to_numerical(store, 'PromoInterval', start_value=0)


        st.write('Stre Data Types:', store.dtypes.astype(str))

        st.write('check for empty (NaN values) for each column again:')
        st.write(store.isnull().sum())



        st.write("We see that we still have NaN values, lets try fixing them with sklean imputer")
        imputer = SimpleImputer().fit(store)
        store_imputed = imputer.transform(store)

        store_new = pd.DataFrame(store_imputed, columns=store.columns.values)
        st.write(store_new.isnull().sum())


        st.write("Results of Store:")
        st.write("Stats:",store.describe(), "First Rows:",store.head(),"Last Rows:", store.tail())





        st.write("All datasets are clean now, so we can start merging them to fit them to our models:")



        st.write("To merge store and train, first we want to check if the \"Store\" column is the same in both datasets:")
        Stores_in_Store = pd.Series(store_new['Store'])
        Stores_in_Train = pd.Series(train['Store'])
        st.write(sum(Stores_in_Train.isin(Stores_in_Store) == False))

        st.write("They are the same, now merge them")
        train_store = pd.merge(train, store_new, how = 'left', on='Store')
        st.write("Train_Store dataset:")
        st.write(train_store.shape)
        st.write(train_store.head())
        st.write(train_store.tail())
        st.write("Check for nulls:")
        st.write(train_store.isnull().sum())

        st.write("Merge Test and Store datasets")
        test_store = test.reset_index().merge(store_new, how = 'left', on='Store')
        st.write("Test Store dataset:")
        st.write(test_store.shape)
        st.write(test_store.head())
        st.write(test_store.tail())
        st.write("Check for nulls:")
        st.write(test_store.isnull().sum())

        st.write("Train Store dataset will be used for training of the model, and test_store for testing.")



        st.session_state.data_type=st.subheader('Visual Exploration:')

        
        st.write("coorelation between columns in train_store")

        fig, ax = plt.subplots()
        sns.heatmap(train_store.corr(), ax=ax)
        st.write(fig)

        st.write("coorelation between columns in test_store")

        fig, ax = plt.subplots()
        sns.heatmap(test_store.corr(), ax=ax)
        st.write(fig)

        st.write("Sales per customer")

        fig, ax = plt.subplots()
        sns.histplot(x="Customers", y="Sales",data=train_store, ax=ax)
        st.write(fig)

        st.write("Sales by year")
        fig, ax = plt.subplots()
        sns.histplot(x="Year", y="Sales",data=train_store, ax=ax)
        st.write(fig)

        fig, ax = plt.subplots()
        sns.boxplot(x="Year", y="Sales",data=train_store, ax=ax)
        st.write(fig)


        st.write("Sales by Month")
        fig, ax = plt.subplots()
        sns.histplot(x="Month", y="Sales",data=train_store, ax=ax)
        st.write(fig)

        fig, ax = plt.subplots()
        sns.boxplot(x="Month", y="Sales",data=train_store, ax=ax)
        st.write(fig)

        
        st.write("Sales on Holidays")
        fig, ax = plt.subplots()
        sns.histplot(x="SchoolHoliday", y="Sales",data=train_store, ax=ax)
        st.write(fig)

        fig, ax = plt.subplots()
        sns.boxplot(x="StateHoliday", y="Sales",data=train_store, ax=ax)
        st.write(fig)
        

        st.write("Now, its time to start modeling. First of all, we will drop columns that are useless for the forecasting, like Customers, Data from train_store, and Date and Id from test_store")
        train_model = train_store.drop(['Customers', 'Date'], axis=1)

        st.write(train_model.head())

        test_model = test_store.drop(['Date','Id'], axis=1)
        st.write(test_model.head())


        st.write("Also, for the train we will drop Sales column, because we want to fit our models without it to make forecast")
        X = train_model.drop('Sales', axis=1)
        y = train_model['Sales']

        st.write("Then, break train test split using \"train_test_split function\"")
        X = train_model.drop('Sales', axis=1)
        y = train_model['Sales']
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


        st.write("The models that we are going to use are: Sklearn's: LinearRegression, Random Forest, GradientBoostingRegressor")
        model_list = {
                'LinearRegression':LinearRegression(),
                'RandomForest':RandomForestRegressor(),
                'GradientBoostingRegressor':GradientBoostingRegressor()
                }

        for  model_name,model in model_list.items():
                st.write(model_name,":")
                model.fit(X_train, y_train)
                st.write("Accuracy:",model.score(X_test, y_test))
                test_model = pd.DataFrame(test_model)
                submission = {}
                submission = pd.DataFrame()
                submission["Predicted Sales"] = model.predict(test_model)
                submission = submission.reset_index()
                st.write(submission)


                
        st.learn("simperasmata:................")                























if st.session_state.workflow == 'Exploratory Data analysis':
        st.session_state.data_type=st.header('**Step 4 - Exploratory Data analysis**')

if st.session_state.workflow == 'Data modeling':
        st.session_state.data_type=st.header('**Step 5 - Data modeling**')

if st.session_state.workflow == 'Visualization & Communication':
        st.session_state.data_type=st.header('**Step 6 - Visualization & Communication**')

if st.session_state.workflow == 'Deployment & Maintenance':
        st.session_state.data_type=st.header('**Step 7 - Deployment & Maintenance**')




about = st.sidebar.expander('About')
about.write('Αυτή η εργασία έγινε στα πλαίσια του μαθήματος **CEI_523 - Επιστήμη Δεδομένων**.')
about.write('**Διδάσκων:** Δρ. Ανδρέας Χριστοφόρου')
about.write('Η ομάδα μας αποτελείται απο τους:')
about.write('👨‍🦱 Στέλιος Μάππουρας')
about.write('👱‍♂️ Ιωάννη Βολονάκη')
about.write('👨‍🦰 Μάριος Κυριακίδης')
about.write('👩‍🦱 Σαββίνα Ρούσου')

helper = st.sidebar.expander('How to use')  
helper.write("This is a helper")            
     

      



            









st.markdown("""
<script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.12.9/umd/popper.min.js" integrity="sha384-ApNbgh9B+Y1QKtv3Rn7W3mgPxhU9K/ScQsAP7hUibX39j7fakFPskvXusvfa0b4Q" crossorigin="anonymous"></script>
<script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
""", unsafe_allow_html=True)