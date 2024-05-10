# Exp: 09 Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the necessary python packages using import statements.

2.Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().

3.Split the dataset using train_test_split.

4.Calculate Y_Pred and accuracy.

5.Print all the outputs.

6.End the Program. 

## Program:
```

Program to implement the SVM For Spam Mail Detection..
Developed by: Kanishka V S
RegisterNumber:  212222230061

```
```py
import chardet
file='spam.csv'
with open (file,'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv("spam.csv",encoding='windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()


x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
## Output:
### Encoding:
![image](https://github.com/kanishka2305/Implementation-of-SVM-For-Spam-Mail-Detection/assets/113497357/70845707-a3e5-4314-b139-19640e523857)

### Head():
![image](https://github.com/kanishka2305/Implementation-of-SVM-For-Spam-Mail-Detection/assets/113497357/ca79eeba-7dce-4a48-a6ce-f7b2d0c2e08e)

### Info():
![image](https://github.com/kanishka2305/Implementation-of-SVM-For-Spam-Mail-Detection/assets/113497357/1ad149c4-2b61-47a9-bb44-fd2266de7a69)

### isnull().sum():
![image](https://github.com/kanishka2305/Implementation-of-SVM-For-Spam-Mail-Detection/assets/113497357/dec4a609-7cd9-45c7-9fc8-a2202621c4e6)

### Prediction of y:
![image](https://github.com/kanishka2305/Implementation-of-SVM-For-Spam-Mail-Detection/assets/113497357/5a6cd90e-2b64-4886-b34d-19e3c8d25a33)

### Accuracy:
![image](https://github.com/kanishka2305/Implementation-of-SVM-For-Spam-Mail-Detection/assets/113497357/f214c3ed-f696-438b-b562-5c2eca122f66)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
