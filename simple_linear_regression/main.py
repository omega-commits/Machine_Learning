import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset=pd.read_csv(r"simple_linear_regression\Salary_Data.csv")
x=dataset.iloc[:, 0:1].values
y=dataset.iloc[:,-1].values



#splitting the dataset into training set and test set
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)

#applying linear regression model
regressor=LinearRegression()
regressor.fit(x_train,y_train)
plt.scatter(x_train,y_train,color="blue")
plt.plot(x_train,regressor.predict(x_train),color="red")
plt.title("salary vs experience")
plt.ylabel("salary")
plt.xlabel("years of experience")
plt.show()
