import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sn

digit=load_digits()
print(digit.data[0])
'''plt.gray()
for i in range(4):
    plt.matshow(digit.images[i])
    plt.show()'''
print(dir(digit))
print(digit.target)
df=pd.DataFrame(digit.data)
print(df.head())
df['target']=digit.target
df.head()
x=df.drop('target',axis='columns')
y=df.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
print(len(x_train))
print(len(x_test))
model=RandomForestClassifier(n_estimators=100)
print(model.fit(x_train,y_train))
print(model.score(x_test,y_test))
y_predit=model.predict(x_test)
print(y_predit)
c=confusion_matrix(y_test,y_predit)
print(c)
plt.figure(figsize=(10,7))
sn.heatmap(c,annot=True)
plt.xlabel('predict')
plt.ylabel('Truth')
plt.show()


