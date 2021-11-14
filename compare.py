import pandas as pd
import datacompy



test_store = pd.read_csv("VolonakisClean/test_store.csv")
clean_test_store = pd.read_csv("clean_test_store.csv")

compare = datacompy.Compare(   

test_store,
clean_test_store,

join_columns=["Id"], #You can also specify a list of columns
abs_tol=0.0001,
rel_tol=0,
df1_name="me",
df2_name="volonakis"

)

print(compare.report())

