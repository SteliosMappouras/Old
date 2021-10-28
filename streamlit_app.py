import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt


st.title('Data Science Project:')
st.caption('Stelios Mappouras\nIoannis Volonakis\nSavvina Rousou\nMarios Kyriakides')


store = pd.read_csv('store.csv')

st.dataframe(store.head())
