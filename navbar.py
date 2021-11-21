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
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


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




st.session_state.workflow = st.sidebar.selectbox('Select a Data Science Life Cycle', ['Business problem', 'Data acquisition', 'Data preparation', 'Exploratory Data analysis', 'Data modeling', 'Visualization & Communication', 'Deployment & Maintenance'] )




if st.session_state.workflow == 'Business problem':
       
        st.session_state.data_type=st.title('''**Πρόβλεψη μελλοντικών πωλήσεων**''')
        st.session_state.data_type=st.header('**Step 1 - Business Problem**') 
        st.session_state.data_type=st.subheader('Πρόβλημα') 
        st.session_state.data_type=st.caption("""Χρησιμοποιώντας τα διαθέσιμα δεδομένα 
        (ιστορικά δεδομένα πωλήσεων) να ακολουθηθούν τα αναγκαία βήματα και 
        αναπτυχθεί ένα μοντέλο όπου θα μπορεί να προβλέψει τις πωλήσεις που 
        μπορούν να γίνουν μελλοντικά (στο σχετικό test dataset).""")

        st.session_state.data_type=st.subheader('Δεδομένα') 
        st.session_state.data_type=st.caption("""Ηistorical data including Sales, Historical data excluding Sales and Supplemental information about the stores """)
           

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
        st.session_state.data_type=st.title('**Step 3 - Data preparation**')
        st.session_state.data_type=st.text("")
        st.session_state.data_type=st.text("")
        st.session_state.data_type=st.text("")

        col1, col2, col3 = st.columns(3)

        with col1:
                st.session_state.data_type=st.markdown("<h5 style='text-align: center;'>Cleaning</h5>", unsafe_allow_html=True) 
                st.session_state.data_type=st.markdown("<h6 style='text-align: center; color:grey'>Inconsistent data types</h6>", unsafe_allow_html=True)
                st.session_state.data_type=st.markdown("<h6 style='text-align: center; color:grey'>Misspelled attributes</h6>", unsafe_allow_html=True)
                st.session_state.data_type=st.markdown("<h6 style='text-align: center; color:grey'>Missing values</h6>", unsafe_allow_html=True)
                st.session_state.data_type=st.markdown("<h6 style='text-align: center; color:grey'>Duplicated values</h6>", unsafe_allow_html=True)   


        with col2:
                st.session_state.data_type=st.markdown("<h5 style='text-align: center;'>Transformation</h5>", unsafe_allow_html=True) 
                st.session_state.data_type=st.markdown("<h6 style='text-align: center; color:grey'>Normalization</h6>", unsafe_allow_html=True)
                st.session_state.data_type=st.markdown("<h6 style='text-align: center; color:grey'>Aggregation</h6>", unsafe_allow_html=True)

        with col3:
                st.session_state.data_type=st.markdown("<h5 style='text-align: center;'>Reduction</h5>", unsafe_allow_html=True) 
                st.session_state.data_type=st.markdown("<h6 style='text-align: center; color:grey'>Reduction in size</h6>", unsafe_allow_html=True)


        st.session_state.data_type=st.text("")
        st.session_state.data_type=st.text("")       
        st.code("""We already started fixing inconsistent data types while importing the datasets, 
we parsed the data types as strings in order to be able to handle it easier later. \n \nAs we saw from visualizing some rows of each dataset, we have some categorical data, 
which we can turn into numerical: \n 
The date can break into Year/Month (we already have day of week in the dataset). \n
Also, for example assortment is alphabetical, \n'a' -> 0, \n'b' -> 1, \netc.""", language="markdown")
       

        code = '''#This function is created to turn categorical column into numerical.
def categorical_to_numerical(df, colname, start_value=0):
        while df[colname].dtype == object:
                myval = start_value # factor starts at "start_value".   
                for sval in df[colname].unique():
                        df.loc[df[colname] == sval, colname] = myval
                        myval += 1
                df[colname] = df[colname].astype(int, copy=False)'''
        st.code(code, language='python')

        st.session_state.data_type=st.text("")


        
        st.session_state.data_type=st.header('Cleaning Train Dataset')
        
        st.code("Step 1 - Check for duplicates and drop them", language="markdown")


       
        st.session_state.data_types=st.write("Shape of Train dataset **before** dropping:",train.shape)
        train = train.drop_duplicates()
        st.session_state.data_types=st.write("Shape of Train dataset **after** dropping:",train.shape)
        st.session_state.data_types=st.write("✔ As we can see, no rows were dropped, we had no duplicates in train dataset.")
        st.session_state.data_type=st.text("")
       
    
        st.code("""Step 2 - Drop stores that are closed because they are useless for our forecast as 
they have no sales""", language="markdown")

        colο1, colο2, colο3 = st.columns(3)

        with colο1:
                st.session_state.data_types=st.write("Before")

        with colο2:
                st.session_state.data_types=st.write("Open Stores:",sum(train['Open'] == 1))

        with colο3:
                st.session_state.data_types=st.write("Closed Stores:",sum(train['Open'] == 0))

        train = train[train.Open != 0]

        colο1, colο2, colο3 = st.columns(3)

        with colο1:
                st.session_state.data_types=st.write("After")

        with colο2:
                st.session_state.data_types=st.write("Open Stores:",sum(train['Open'] == 1))

        with colο3:
                st.session_state.data_types=st.write("Closed Stores:",sum(train['Open'] == 0))

        st.session_state.data_type=st.text("")

        st.code("""Step 3 - We want to break the Date column into Year and Month, then drop date.""", language="markdown")

        train['Year'] = pd.DatetimeIndex(train['Date']).year
        train['Month'] = pd.DatetimeIndex(train['Date']).month

        sample = train.head(200)
        st.write(sample)
        st.session_state.data_type=st.text("")

        st.code("""Step 4 - We want to convert the remaining categorical data into numerical.
We use our function categorical_to_numerical that we mentioned above, for 
StateHoliday and SchoolHoliday.
""", language="markdown")

        
        st.session_state.data_type=st.text("")

        col1, col2, col3 = st.columns(3)

        with col1:
               st.write('', train.dtypes.astype(str))
        with col2:
                categorical_to_numerical(train,'StateHoliday')
                categorical_to_numerical(train,'SchoolHoliday')
                st.write('', train.dtypes.astype(str))
                st.write('Train data types **after**')

        st.session_state.data_type=st.text("")
        st.code("""Step 5 - Check for empty (NaN values) for each column""", language="markdown")
        st.write(train.isnull().sum())
        st.session_state.data_types=st.write("✔ All values are zero, we have no empty values.")
        st.session_state.data_type=st.text("")


        st.code("""Step 6 - Specify the columns that are going to be used on the model""", language="markdown")

        #we have no NANs in train dataset
        #select columns for the training data
        train = train[['Store', 'DayOfWeek', 'Date', 'Year', 'Month', 'Customers', 'Open','Promo', 'StateHoliday', 'SchoolHoliday', 'Sales']]
        st.write(list(train.columns.values))
        st.session_state.data_type=st.text("")
        st.session_state.data_type=st.text("")


        st.write("**Results of Train:** \n"," \n **Stats:**",train.describe(), "**First Rows:**",train.head(),"**Last Rows:**", train.tail())

        st.session_state.data_type=st.text("")
        st.session_state.data_type=st.text("")





        st.session_state.data_type=st.header('Cleaning Test Dataset')
        st.session_state.data_types=st.write('Repeat the same preparation process')

        st.code("Step 1 - Check for duplicates and drop them", language="markdown")
        st.session_state.data_types=st.write("Shape of Test dataset **before** dropping:",test.shape)
        test = test.drop_duplicates()
        st.session_state.data_types=st.write("Shape of Test dataset **after** dropping:",test.shape)
        st.session_state.data_types=st.write("✔ As we can see, no rows were dropped, we had no duplicates in train dataset.")
        st.session_state.data_type=st.text("")

        st.code("""Step 2 - Drop the stores that are closed because they are useless for our forecast as 
they have no sales""", language="markdown")
        
        colο1, colο2, colο3 = st.columns(3)

        with colο1:
                st.session_state.data_types=st.write("Before")

        with colο2:
                st.session_state.data_types=st.write("Open Stores:",sum(test['Open'] == 1))

        with colο3:
                st.session_state.data_types=st.write("Closed Stores:",sum(test['Open'] == 0))

        test = test[test.Open != 0]

        colο1, colο2, colο3 = st.columns(3)

        with colο1:
                st.session_state.data_types=st.write("After")

        with colο2:
                st.session_state.data_types=st.write("Open Stores:",sum(test['Open'] == 1))

        with colο3:
                st.session_state.data_types=st.write("Closed Stores:",sum(test['Open'] == 0))

        st.session_state.data_type=st.text("")

        st.code("""Step 3 - We want to break the Date column into Year and Month, then drop date.""", language="markdown")

        test['Year'] = pd.DatetimeIndex(test['Date']).year
        test['Month'] = pd.DatetimeIndex(test['Date']).month

        test.drop(columns=['Date'])
        sample = test.head(200)
        st.write(sample)
        st.session_state.data_type=st.text("")

        st.code("""Step 4 - We want to convert the remaining categorical data into numerical""", language="markdown")
        
        st.session_state.data_type=st.text("")

        col1, col2, col3 = st.columns(3)

        with col1:
                st.write('', test.dtypes.astype(str))

        with col2:
                categorical_to_numerical(test,'StateHoliday')
                categorical_to_numerical(test,'SchoolHoliday')
                categorical_to_numerical(test,'Open')
                st.write('', test.dtypes.astype(str))
                st.write('Test data types **after**')
        
        
        st.session_state.data_type=st.text("")
        
        st.code("""Step 6 - Check for empty (NaN values) for each column""", language="markdown")
        st.write(test.isnull().sum())

        st.code("""There are 11 missing values in Open column.
Step 7 - Check from which store they come from""", language="markdown")
        st.write(test[np.isnan(test['Open'])])
        st.session_state.data_type=st.text("")

        st.code("""It is from the store 622.
Step 8 - Search if there is any info about this store in train dataset""", language="markdown")
        st.write(train[train['Store'] == 622].head())
        st.write('We found some information, so we will assume that the store is open.')
        test[np.isnan(test['Open'])] = 1

        
        st.code("""Step 9 - Check if there are still missing values""", language="markdown")
        st.write(test.isnull().sum())

        st.session_state.data_type=st.text("")
        st.session_state.data_type=st.text("")

        st.code("""Step 10 - Specify the columns that are going to be used on the model and change 
the order to match train dataset.""", language="markdown")

        col1, col2, col3 = st.columns(3)

        with col1:
                st.write('Test Data Types:', test.dtypes.astype(str))

        with col2:
                #select columns for the testing data
                test = test[['Store', 'DayOfWeek', 'Date', 'Year', 'Month', 'Open','Promo', 'StateHoliday', 'SchoolHoliday']]
                st.write(list(test.columns.values))
        
        
        st.session_state.data_type=st.text("")
        st.session_state.data_type=st.text("")
        st.write("**Results of Test:**")
        st.write("**Stats:**",test.describe(), "**First Rows:**",test.head(),"**Last Rows:**", test.tail())

        st.session_state.data_type=st.text("")
        st.session_state.data_type=st.text("")
        st.session_state.data_type=st.text("")



        
        
        st.session_state.data_type=st.header('Cleaning Store Dataset')
        st.session_state.data_types=st.write('Repeat the same preparation process one last time')
        st.session_state.data_type=st.text("")

        st.code("Step 1 - Check for duplicates and drop them", language="markdown")
        st.session_state.data_types=st.write("Shape of Store dataset **before** dropping:",store.shape)
        store = store.drop_duplicates()
        st.session_state.data_types=st.write("Shape of Store dataset **after** dropping:",store.shape)
        st.session_state.data_types=st.write("✔ As we can see, no rows were dropped, we had no duplicates in train dataset.")
        st.session_state.data_type=st.text("")

        st.code("""Step 2 - Check for empty (NaN values) for each column""", language="markdown")
        st.write(store.isnull().sum())



        st.write(store.describe())
        st.code("""Promo related values are coorelated. If there is no promo, coorelated values should 
be zeros.
As we can see from the missing values, we have this problem in our dataset. \n
Step 3 - Fix coorelated values""", language="markdown")
        
        store.loc[store['Promo2'] == 0, ['Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval']] = 0
        store.loc[store['Promo2'] != 0, 'Promo2SinceWeek'] = store['Promo2SinceWeek'].max() - store.loc[store['Promo2'] != 0, 'Promo2SinceWeek']
        store.loc[store['Promo2'] != 0, 'Promo2SinceYear'] = store['Promo2SinceYear'].max() - store.loc[store['Promo2'] != 0, 'Promo2SinceYear']

        st.write(store.describe())

        st.code("""Step 4 - We want to convert the remaining categorical data into numerical
We use our function categorical_to_numerical for StoreType, assortment and 
Promo Interval.
""", language="markdown")
        
        
        st.session_state.data_type=st.text("")

        col1, col2, col3 = st.columns(3)

        with col1:
                st.write('', store.dtypes.astype(str))

        with col2:
                categorical_to_numerical(store, 'StoreType')
                categorical_to_numerical(store, 'Assortment')
                store['PromoInterval'].unique()
                categorical_to_numerical(store, 'PromoInterval', start_value=0)
                st.write('', store.dtypes.astype(str))
                st.write('Store data types **after**')
        
        st.session_state.data_type=st.text("")

        st.code("""Step 6 - Check for empty (NaN values) for each column again""", language="markdown")
        st.write(store.isnull().sum())


        st.code("""We see that we still have NaN values.
Step 7 - Fix them with sklean imputer""", language="markdown")
        imputer = SimpleImputer().fit(store)
        store_imputed = imputer.transform(store)

        store_new = pd.DataFrame(store_imputed, columns=store.columns.values)
        st.write(store_new.isnull().sum())

        st.session_state.data_type=st.text("")
        st.session_state.data_type=st.text("")

        st.write("**Results of Store:**")
        st.write("**Stats:**",store.describe(), "**First Rows:**",store.head(),"**Last Rows:**", store.tail())

        st.session_state.data_type=st.text("")
        st.session_state.data_type=st.text("")



        st.success('All datasets are clean now, so we can start merging them to fit them to our models.')


        
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