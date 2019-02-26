# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 10:19:45 2019

@author: saksh
"""



#Objective is to detemnine if price of diamond is greater than 3000$ or not

#Importing necessary packages
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


#import the data file
diamond= pd.read_excel("C:/Users/saksh/Downloads/CoursesSpring19/STAT 656/Python_Projects/Week 2/diamondswmissing.xlsx")
#Information about dataset
print(diamond.shape)
print(diamond.columns)
print(diamond.head(5))

#Encoding price as binary variable since our objective is classification
diamond['price'] = [0 if x<3000 else 1 for x in diamond['price']]
print(diamond['price'].value_counts())
#so we can see that 30334 diamonds have price less than $3000 and 23606 have price more than $3000

# Segregating response variable from predictors
X=diamond.drop('price',1)
Y=diamond['price'] 

#Replacing outliers with missing values
diamond.loc[(diamond.Carat>5.5) | (diamond.Carat<0.2),'Carat']=np.nan
diamond.loc[(diamond.depth>80) | (diamond.depth<40),'depth']=np.nan
diamond.loc[(diamond.table>100) | (diamond.table<40),'table']=np.nan
diamond.loc[(diamond.x>11) | (diamond.x<0),'x']=np.nan
diamond.loc[(diamond.y>32) | (diamond.y<0),'y']=np.nan
diamond.loc[(diamond.z>60) | (diamond.z<0),'z']=np.nan

#Imputing missing values
print(Y.isnull().sum()) #No missing values in response
print(X.isnull().sum())
X.fillna(X.mean(), inplace=True)

#Encoding categoical variables
dummy_list=['cut', 'color', 'clarity']
for i in dummy_list:
    dummy=pd.get_dummies(X[i], prefix=i)
    X=X.drop(i,1)  #dropping off original columns
    X=pd.concat([X,dummy], axis=1) #concatinating new encoded columns in dataframe

#Now our data is free from any ouliers or missing values
print(X.isnull().sum())

#Spliting the data set in 10 folds for validation
kfold = model_selection.KFold(n_splits=10, random_state=7)

#Fitting logistic regression model
lgr = LogisticRegression()
lgr.fit(X,Y)
lgr_10_scores = cross_val_score(lgr, X, Y, cv=kfold)
print("\nAccuracy Scores by Fold: ", lgr_10_scores)
mean_lgr=lgr_10_scores.mean()

#linear discriminant analysis
lda=LinearDiscriminantAnalysis()
lda.fit(X,Y)
lda_10_scores = cross_val_score(lda, X, Y, cv=kfold)
print("\nAccuracy Scores by Fold: ", lda_10_scores)
mean_lda=lda_10_scores.mean()

#K- nearest neighbours
knn=KNeighborsClassifier(n_neighbors=100)
knn.fit(X,Y)
knn_10_scores = cross_val_score(knn, X, Y, cv=kfold)
print("\nAccuracy Scores by Fold: ", knn_10_scores)
mean_knn=knn_10_scores.mean()

#Decision tree classifier
dtc=DecisionTreeClassifier()
dtc.fit(X,Y)
dtc_10_scores = cross_val_score(dtc, X, Y, cv=kfold)
print("\nAccuracy Scores by Fold: ", dtc_10_scores)
mean_dtc=dtc_10_scores.mean()

#Model Assessment 
models=['lgr','lda','knn','dtc']
accuracy=[mean_lgr, mean_lda,mean_knn,mean_dtc]
print(models ,"\n" , accuracy)


#Displaying the best model by comparing their mean accuracy
i=accuracy.index(max(accuracy))
best_model=models[i]
print("The best model is {0}".format(best_model))

#The best model obtained is decision tree classifier

#Fitting the model for best mthod using 70% training data and 30% test data
X_train, X_validate, Y_train, Y_validate =train_test_split(X,Y,test_size = 0.3, random_state=7)
dtc1 = DecisionTreeClassifier(criterion='gini',max_depth=5,min_samples_split=5,min_samples_leaf=5)
dtc1 = dtc1.fit(X_train,Y_train)
#Accuracy of model
print(dtc1.score(X_validate,Y_validate))
#Classifying first 30 observation of dataset
predictions=dtc1.predict(X_validate)
print(diamond.head(30), predictions[0:30])








    
