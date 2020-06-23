# Exploratory Data Analysis- Considering all built in libraries of Matplotlib and Seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv("E:\\ExcelR\\titanic_train.csv")

# Step-1 Checking NUll Values


 # There are null values in age and cabin
sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap="viridis")

#Step -2 Understanding relationship between Variables using Seaborn graphs

data.head()
data.columns

# P-class relation with Survival Rate

sns.countplot("Pclass",hue="Survived",data=data,color="red")

# Value count of P-class
data["Pclass"].value_counts()

# Sex relationship with survival Rate

sns.countplot("Sex",hue="Survived",data=data)

# Age

sns.distplot(data["Age"].dropna(),kde=True)

# Step-3 Data Cleaning

# if there is a relationship between age and class

sns.boxplot("Pclass","Age",data=data)

# 1- Class- 39, 2- Class- 29 3- Class- 24


def remnul(x):
    Age=x[0]
    Pclass=x[1]
    if pd.isnull(Age):
        if Pclass==1:
            return 37
        elif Pclass==2:
            return 29
        else:
            return 24
        
    else:
        return(Age)

data["Age"]=data[["Age","Pclass"]].apply(remnul,axis=1)
        
data["Age"].isnull().sum()

# Step 4- Creating Dummies

sex=pd.get_dummies(data["Sex"],drop_first=True)
Embarked=pd.get_dummies(data["Embarked"],drop_first=True)

# Step 5 Removing the Unnecessary Columns and adding the newely created Dummies
data.columns
data=data.drop(["PassengerId","Name","Sex","Ticket","Fare","Embarked"],axis=1)
data=pd.concat([data,sex["male"],Embarked["Q"],Embarked["S"]],axis=1)

# Removing Cabin

data=data.drop("Cabin",axis=1)

data.isnull().sum()

#Step 6 Applying Classification Algorithm

X=data.drop(["Survived"],axis=1)
Y=data["Survived"]

# Spliting the data into train and test

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.30)

from sklearn.linear_model import LogisticRegressionCV
lrcv=LogisticRegressionCV()
Model=lrcv.fit(X_train,Y_train)
Y_pred=Model.predict(X_test)

# Checking the Accuracy of the Model

from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
print(classification_report(Y_test,Y_pred))
print(accuracy_score(Y_test,Y_pred))
print(confusion_matrix(Y_test,Y_pred))

