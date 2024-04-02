# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries.

2.Upload and read the dataset.

3.Check for any null values using the isnull() function.

4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

5.Find the accuracy of the model and predict the required values by importing the required module from sklearn. . 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: KALPANA S
RegisterNumber: 212222040069 
*/
import pandas as pd
data=pd.read_csv("Employee.csv")

data.head()

data.info()

data.isnull().sum()

data['left'].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["salary"]=le.fit_transform(data["salary"])
data.head()

x = data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y = data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion = "entropy")
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
print(accuracy)

dt.predict([[0.5,0.8,9,260,6,0,1,2]])

```

## Output:
### DATASET
![image](https://github.com/22008650/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/122548204/34cb16f1-7bcd-4741-891c-99c77f373c54)
### data.info()
![image](https://github.com/22008650/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/122548204/fdb5b932-e85c-4275-b299-76ada5228790)
### CHECKING IF NULL VALUES ARE PRESENT
![image](https://github.com/22008650/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/122548204/4fd0df3d-dbfc-409a-8d27-95a62445de0e)
### VALUE_COUNTS()
![image](https://github.com/22008650/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/122548204/04b32f6f-a872-4de4-b0f4-a03e464932f0)
### DATASET AFTER ENCODING
![image](https://github.com/22008650/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/122548204/26063147-f564-46b2-9e6d-b7f2e92ee594)
### X-VALUES
![image](https://github.com/22008650/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/122548204/f7f0756b-ebce-4b19-81bd-dc54dde836d6)
### ACCURACY
![image](https://github.com/22008650/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/122548204/8b1cace9-038c-4225-9ea8-2fd5cd4b749e)
### dt.predict()
![image](https://github.com/22008650/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/122548204/afefdadd-5df7-485d-a0c6-9994f3e818c5)




## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
