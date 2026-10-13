import pandas as pd 
df =pd.read_csv(r"C:\Users\Mr Computer\Desktop\web developement code\heart_disease.csv" )  
print(df)
print(df.describe())
print(df.info())
print(df.shape)
df['Heart Disease Status'] = df['Heart Disease Status'].map({'Yes': 1, 'No': 0})
print(df)