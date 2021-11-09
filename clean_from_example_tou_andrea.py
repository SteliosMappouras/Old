import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import sklearn as sk

#load all datasets
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
store = pd.read_csv("store.csv")

#extract time related data
train['Date'] = pd.to_datetime(train['Date'], errors='coerce')

train['Year'] = train['Date'].dt.year
train['Month'] = train['Date'].dt.month
train['Day'] = train['Date'].dt.day
train['n_days'] = (train['Date'].dt.date-train['Date'].dt.date.min()).apply(lambda x:x.days)


train = train.merge(store, on='Store', how='left')

print(train.head())
