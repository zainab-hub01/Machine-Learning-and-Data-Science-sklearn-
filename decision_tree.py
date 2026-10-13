import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
model=tree.DecisionTreeClassifier()


df =pd.read_csv(r"C:\Users\Mr Computer\Desktop\web developement code\heart_disease.csv" )  #gcsfs cloud sa data read karna 

print(df)
print(df.describe())
print(df.info())
print(df.shape)
df['Heart Disease Status'] = df['Heart Disease Status'].map({'Yes': 1, 'No': 0})
print(df)
print(df.isnull().sum())
c=df.select_dtypes(include=['number'])
m=print(c.corr())
le_gender=LabelEncoder()
le_cholesterol_level=LabelEncoder()
le_age=LabelEncoder()
le_fasting_blood_sugar=LabelEncoder()
le_CRP_level=LabelEncoder()
df['Gender_n'] = le_gender.fit_transform(df['Gender'])
df['Cholesterol level_n'] = le_cholesterol_level.fit_transform(df['Cholesterol Level'])
df['Age_n'] = le_age.fit_transform(df['Age'])
df['fasting blood sugar_n'] = le_fasting_blood_sugar.fit_transform(df['Fasting Blood Sugar'])
df['CRP level_n'] = le_CRP_level.fit_transform(df['CRP Level'])
print(df)
target=df['Heart Disease Status']
print(target)
t_n=df.drop(['Fasting Blood Sugar','CRP Level','Age','Cholesterol Level','Gender'],axis='columns',)
print(t_n)
X = df[['Gender_n', 'Cholesterol level_n', 'Age_n', 'fasting blood sugar_n', 'CRP level_n']]
y = df['Heart Disease Status']

# Ab model ko fit karein
model.fit(X, y)
model.score(X,y)
prediction=model.predict([[1,66,28,12,8468]])
print(prediction)
#pair=c.unstack()
#sort=pair.abs().sort_values(ascending=False)
#filter=sort[sort<1.0].drop_duplicates()
##print(filter.head(5))
#print(df.columns.to_list())    # colum name
##numeric_df = df.select_dtypes(include=["number"])
#numeric_df.corr().plot(kind="bar", figsize=(12, 6))
#plt.title("Correlation Bar Chart")
##plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
#plt.show()
#input=df.drop( 'Heart Disease Status',axis="columns")
#print(input)
#target=df(['Heart Disease Status'])
#print(target)


