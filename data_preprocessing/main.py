#importing libraries and datset
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
dataset=pd.read_csv(r"data_preprocessing\Data.csv")
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

#Taking care of the missing values

imputer=SimpleImputer(missing_values=np.nan,strategy="mean")
imputer.fit(x[:,1:3])
imputer.transform(x[:,1:3])
x[:,1:3]=imputer.transform(x[:,1:3])

#Encoding categorical data

#Encoding the independent variable
CT=ColumnTransformer(transformers=[("encoder",OneHotEncoder(),[0])],remainder="passthrough")
x=np.array(CT.fit_transform(x))


#Encoding the dependent variable
LE=LabelEncoder()
y=LE.fit_transform(y)
print(y)

#splitting the dataset into test set and training set

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)

#Feature scalling

SC=StandardScaler()
x_train[:,3:]=SC.fit_transform(x_train[:,3:])
x_test[:,3:]=SC.fit_transform(x_test[:,3:])

print(x_train)
print(y_train)




