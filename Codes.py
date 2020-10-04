# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 11:00:35 2020

@author: Prachi Palvi 75291019025
"""
#%%
"""
Shill Bidding Dataset Data Set

-Shill bidding is when someone bids 
on an item to artificially increase its price, 
desirability, or search standing.

-could create an unfair advantage, 
or cause another bidder to pay more than they should.

- The data set we have is labelled data set, we can use 
(Supervised Learning) classification techniques to identify fraudulent shill bidders
with - abnormal bidding behaviour or otherwise the normal bidders

- we will be using three classification models 
  1) Naive Bayes 
  2) Logistic Regression
  3) Decesion Tree classifier
"""
#%%
#importing required libraries 

import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,auc,roc_auc_score
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

#%% Exploratory data analysis

#importing data set 

SBdata=pd.read_csv("C:\\Users\\Admin\\Desktop\\Msc ASA sem 3\\SC 5\\ICA exam\\Shill_Bidding_Dataset.csv")
SBdata.info
SBdata.shape  # the dimensions of data set 
""" 
from this we know the first three variables are unique identification variables
are categorical and of no use for our model.

The class variable is our target variable(dependent) and variables 4 - 12 are 
independent variables contributing to the class variable.

"""
# removing unnecessary categorical variables
SBdata1=SBdata.drop(['Record_ID','Auction_ID','Bidder_ID'],axis=1)

summary=SBdata1.describe() #summary

" summary tells us that all variables are floats except for Auction_Duration "

# lets check the ratio of frauds v/s normal bidders 

print("Class as pie chart:")
fig, ax = plt.subplots(1, 1)
ax.pie(SBdata1.Class.value_counts(),autopct='%1.1f%%', labels=['Genuine','Fraud'], colors=['yellowgreen','r'])
plt.axis('equal')
plt.ylabel('')

# so there are 10.7 % fraud bidders 

#plot of variables to see the variable wise significant difference for genuine and fraud bidders

print("Bidder Tendency")
fig, (ax1, ax2) = plt.subplots(2, 1, sharex = True, figsize=(6,3))
ax1.hist(SBdata1.Bidder_Tendency[SBdata1.Class==0],bins=48,color='g',alpha=0.5)
ax1.set_title('Genuine')
ax2.hist(SBdata1.Bidder_Tendency[SBdata1.Class==1],bins=48,color='r',alpha=0.5)
ax2.set_title('Fraud')
plt.xlabel('Bidder Tendency')
plt.ylabel('Bids')

"""
the above plot shows that genuine bidder tendency ranges from 0 to 0.2
while a fraud ranges 0 to 1

"""
print("Bidding Ratio")
fig, (ax1, ax2) = plt.subplots(2, 1, sharex = True, figsize=(6,3))
ax1.hist(SBdata1.Bidding_Ratio[SBdata1.Class==0],bins=48,color='g',alpha=0.5)
ax1.set_title('Genuine')
ax2.hist(SBdata1.Bidding_Ratio[SBdata1.Class==1],bins=48,color='r',alpha=0.5)
ax2.set_title('Fraud')
plt.xlabel('Bidding Ratio')
plt.ylabel('Bids')

"""
the above plot shows that genuine bidding ranges from 0 to 0.2
while a fraud ranges 0.1 to 0.8

"""
print("Successive Outbidding")
fig, (ax1, ax2) = plt.subplots(2, 1, sharex = True, figsize=(6,3))
ax1.hist(SBdata1.Successive_Outbidding[SBdata1.Class==0],bins=48,color='g',alpha=0.5)
ax1.set_title('Genuine')
ax2.hist(SBdata1.Successive_Outbidding[SBdata1.Class==1],bins=48,color='r',alpha=0.5)
ax2.set_title('Fraud')
plt.xlabel('Sucessive Outbidding')
plt.ylabel('Bids')

"""
the discription of variable tells us that a shill bidder succesively outbids himself
even though he is the current winner to increase the price gradually with small consecutive increments

so from the plot we can see a fraud/shill bidder will have successive outbidding value 0.5 or 1.0
where as a genuine bidder will always have 0 

"""
print("Last bidding")
fig, (ax1, ax2) = plt.subplots(2, 1, sharex = True, figsize=(6,3))
ax1.hist(SBdata1.Last_Bidding[SBdata1.Class==0],bins=48,color='g',alpha=0.5)
ax1.set_title('Genuine')
ax2.hist(SBdata1.Last_Bidding[SBdata1.Class==1],bins=48,color='r',alpha=0.5)
ax2.set_title('Fraud')
plt.xlabel('Last bidding')
plt.ylabel('Bids')  ##### similar results 

print("Auction_Bids")
fig, (ax1, ax2) = plt.subplots(2, 1, sharex = True, figsize=(6,3))
ax1.hist(SBdata1.Auction_Bids[SBdata1.Class==0],bins=48,color='g',alpha=0.5)
ax1.set_title('Genuine')
ax2.hist(SBdata1.Auction_Bids[SBdata1.Class==1],bins=48,color='r',alpha=0.5)
ax2.set_title('Fraud')
plt.xlabel('Auction_Bids')
plt.ylabel('Bids') ##### similar results 


print("Auction Starting Price")
fig, (ax1, ax2) = plt.subplots(2, 1, sharex = True, figsize=(6,3))
ax1.hist(SBdata1.Starting_Price_Average[SBdata1.Class==0],bins=48,color='g',alpha=0.5)
ax1.set_title('Genuine')
ax2.hist(SBdata1.Starting_Price_Average[SBdata1.Class==1],bins=48,color='r',alpha=0.5)
ax2.set_title('Fraud')
plt.xlabel('Starting_Price_Average')
plt.ylabel('Bids') ##### similar results 

print("Early Bidding")
fig, (ax1, ax2) = plt.subplots(2, 1, sharex = True, figsize=(6,3))
ax1.hist(SBdata1.Early_Bidding[SBdata1.Class==0],bins=48,color='g',alpha=0.5)
ax1.set_title('Genuine')
ax2.hist(SBdata1.Early_Bidding[SBdata1.Class==1],bins=48,color='r',alpha=0.5)
ax2.set_title('Fraud')
plt.xlabel('Early Bidding')
plt.ylabel('Bids') ##### similar results 

print("Winning_Ratio")
fig, (ax1, ax2) = plt.subplots(2, 1, sharex = True, figsize=(6,3))
ax1.hist(SBdata1.Winning_Ratio[SBdata1.Class==0],bins=48,color='g',alpha=0.5)
ax1.set_title('Genuine')
ax2.hist(SBdata1.Winning_Ratio[SBdata1.Class==1],bins=48,color='r',alpha=0.5)
ax2.set_title('Fraud')
plt.xlabel('Winning_Ratio')
plt.ylabel('Bids')

"""
so a normal behaviour bidding has majority winning ratio value 0 where has 
fraud would have winning ratio ranging from 0.7 to 1

"""

print("Auction_Duration")
fig, (ax1, ax2) = plt.subplots(2, 1, sharex = True, figsize=(6,3))
ax1.hist(SBdata1.Auction_Duration[SBdata1.Class==0],bins=48,color='g',alpha=0.5)
ax1.set_title('Genuine')
ax2.hist(SBdata1.Auction_Duration[SBdata1.Class==1],bins=48,color='r',alpha=0.5)
ax2.set_title('Fraud')
plt.xlabel('Auction_Duration')
plt.ylabel('Bids') ##### similar results

"""
so the conclusion is the variables -
Bidder Tendency, Bidding Ratio, Successive Outbidding, and winning ratio,
are significantly different for the target variable class(0 and 1).

where as the distrubution of other variables is same 
for both the class types(0 - normal bidding behavior and 1 - fraud)

"""
#the distribution can be visualized using this aswell

import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(28, 1)
plt.figure(figsize=(6,28*4))
for i, col in enumerate(SBdata1[SBdata1.iloc[:,0:9].columns]):
    ax5 = plt.subplot(gs[i])
    sns.distplot(SBdata1[col][SBdata1.Class == 1], bins=50, color='r')
    sns.distplot(SBdata1[col][SBdata1.Class == 0], bins=50, color='g')
    ax5.set_xlabel('')
    ax5.set_title('feature: ' + str(col))
plt.show()

#%% model building

# so we futher train a model specifically on the significant variables 

SBdata2=SBdata1[['Bidder_Tendency','Bidding_Ratio','Successive_Outbidding','Winning_Ratio','Class']]
SBdata2.head

#now we split the data into train and test sets

y = SBdata2['Class'].values #target
X = SBdata2.drop(['Class'],axis=1).values #features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42, stratify=y)
                                                        
print("train-set size: ", len(y_train),
      "\ntest-set size: ", len(y_test))
print("fraud cases in test-set: ", sum(y_test))
X_train
X_test
y_train
y_test

#%%

## Creating a pipeline function for common classifier algorithm

def get_predictions(clf, X_train, y_train, X_test):
    # create classifier
    clf = clf
    # fit it to training data
    clf.fit(X_train,y_train)
    # predict using test data
    y_pred = clf.predict(X_test)
    # Compute predicted probabilities: y_pred_prob
    y_pred_prob = clf.predict_proba(X_test)
    #for fun: train-set predictions
    train_pred = clf.predict(X_train)
    print('train-set confusion matrix:\n', confusion_matrix(y_train,train_pred)) 
    return y_pred, y_pred_prob

## function to get classifiers score 

def print_scores(y_test,y_pred,y_pred_prob):
    print('test-set confusion matrix:\n', confusion_matrix(y_test,y_pred)) 
    print("recall score: ", recall_score(y_test,y_pred))
    print("precision score: ", precision_score(y_test,y_pred))
    print("f1 score: ", f1_score(y_test,y_pred))
    print("accuracy score: ", accuracy_score(y_test,y_pred))
    print("ROC AUC: {}".format(roc_auc_score(y_test, y_pred_prob[:,1])))

#%%

# training a naive bayes model for classification 
y_pred, y_pred_prob = get_predictions(GaussianNB(), X_train, y_train, X_test)

print_scores(y_test,y_pred,y_pred_prob)

# Accuracy = 96.91 %


# hence we can see that the model has correclty classified all the 135 values as frauds/ shill bidders

#%%
# training a logistic regression model 
y_pred, y_pred_prob = get_predictions(LogisticRegression(C = 0.01, penalty = 'l1'), X_train, y_train, X_test)

print_scores(y_test,y_pred,y_pred_prob)

# Accuracy = 96.28 %
# Recall Score is also low 0.666


# how ever it is not the same for this LR model, further that can be improvised using undersampled data


#%% 
" Decision Tree Classifier "

# training Decision tree 
y_pred, y_pred_prob = get_predictions(DecisionTreeClassifier(max_depth=4), X_train, y_train, X_test) 
# by using max_depth=4 we perform pre pruning, highest accuracy at 4

print_scores(y_test,y_pred,y_pred_prob)

# hence Decesion tree classifier has the heighest accuracy of 98.33 %

# we can visualize a decision tree using the following code.

clf = DecisionTreeClassifier()
clf2=clf.fit(X_test,y_pred)
tree.plot_tree(clf2)

#%%
""" CONCLUSION 

It is important to classify or predict which bidder is a shill bidder and is influencing 
bidding prices in a fraudulent way. Hence we built these models using a labelled data set.

The data set was properly analysed and checked for trends, missing values and 
unnecessary variables and was cleaned accordingly before modelling.

We trained models for Naive bayes, Logistic Regression and Decesion Tree classifiers.

The accuracy for Decesion Tree classifier model is highest hence 
it would be best to use for future predictions.

Further using this data we can try and learn Unsupervised learning methods like 
clustering for labelling the data set.

"""

#%% Thank you!



