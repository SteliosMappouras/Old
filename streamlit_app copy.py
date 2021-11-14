import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt


#st.title('Data Science Project:')
#st.caption('Stelios Mappouras\nIoannis Volonakis\nSavvina Rousou\nMarios Kyriakides')


store = pd.read_csv('store.csv')

#get row, column count
size = store.shape

#seperate numeric with categorical columns
#get numeric
store_numeric = store.select_dtypes(include=[np.number])
numeric_cols = store_numeric.columns.values

#get categorical
store_categorical = store.select_dtypes(exclude=[np.number])
categorical_cols = store_categorical.columns.values

print("dataset size:", size)
print("numeric ", numeric_cols)
print("categorical: ", categorical_cols)


#calculate and plot percentage of values missing from each column then store the info in a dataframe
values_list = list()
cols_list=list()

for col in store.columns:
    missing=np.mean(store[col].isnull())*100
    cols_list.append(col)
    values_list.append(missing)
missing_df = pd.DataFrame()
missing_df['col'] = cols_list
missing_df['missing'] = values_list

missing_df.loc[missing_df.missing > 0].plot(kind='bar', figsize=(12,8))

info = store.info()
print(info)


##plt.show()


print(store.isnull().sum())