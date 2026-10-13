import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
iris=load_iris()
print(iris)
len(iris.data)
df=pd.DataFrame(iris.data,columns=iris.feature_names)
print(df)
df['target']=iris.target
df.head()
print(iris.target)
print(df[df.target==2].head())
df['flower_name']=df.target.apply(lambda x:iris.target_names[x])
print(df.head())
print([df[45:55]])
df0=df[:50]
df1=df[50:100]
df2=df[100:]
'''plt.xlabel('sepal length')

plt.xlabel('sepal width')
plt.scatter(df0['sepal length (cm)'],df0['sepal width (cm)'],color='red',marker='+')
plt.scatter(df1['sepal length (cm)'],df1['sepal width (cm)'],color='blue',marker='+')
plt.scatter(df2['sepal length (cm)'],df2['sepal width (cm)'],color='green',marker='+')'''
x=df.drop(['target','flower_name'],axis='columns')
y=df.target

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
print(len(x_train))
print(len(x_test))
model=SVC()
print(model.fit(x_train,y_train))
print(model.score(x_test,y_test))
#print(model.predict[5.8,2.7,5.1,1.9])
model_g=SVC(gamma=15)
print(model_g.fit(x_train,y_train))
print(model_g.score(x_test,y_test))
model_linear=SVC(kernel='linear')
print(model_linear.fit(x_train,y_train))
print(model_linear.score(x_test,y_test))





plt.xlabel('petal length')
plt.xlabel('petal width')
plt.scatter(df0['petal length (cm)'],df0['petal width (cm)'],color='red',marker='+')
plt.scatter(df1['petal length (cm)'],df1['petal width (cm)'],color='blue',marker='+')
plt.scatter(df2['petal length (cm)'],df2['petal width (cm)'],color='green',marker='+')
plt.show()