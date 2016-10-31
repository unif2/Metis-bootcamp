# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 17:16:55 2016

@author: avasilye
"""

import pandas as pd
import numpy as np
import sklearn.linear_model as lm
import statsmodels.api as sm 
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.learning_curve import learning_curve
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.grid_search import GridSearchCV
from sklearn import linear_model
import matplotlib.pyplot as plt
import seaborn

df15 = pd.read_csv("2015 CHR Analytic Data.csv")

# decide which features are usefuly for this analysis
columns = ["STATECODE", "COUNTYCODE", "State", "County", "Adult smoking Value", "Adult obesity Value", "Physical inactivity Value", 
           "Excessive drinking Value", "Median household income Value","Alcohol-impaired driving deaths Value", 
           "Drug poisoning deaths Value","Poor mental health days Value", "Premature age-adjusted mortality Value",
           "Preventable hospital stays Value","Poor or fair health Value", "Poor physical health days Value"]
df15_new = df15[columns]

# divide up states by regions to assign them different median incomes
South  = ["Delaware", "District of Columbia", "Florida", "Georgia", "Maryland", 
          "North Carolina", "South Carolina", "Virginia","West Virginia","Alabama", 
          "Kentucky", "Mississippi", "Tennessee","Arkansas", "Louisiana", "Oklahoma","Texas"]
North = ["Connecticut", "Maine", "Massachusetts", "New Hampshire", "Rhode Island","Vermont"
        "New Jersey", "New York", "Pennsylvania"]
Mid = ["Illinois", "Indiana", "Michigan", "Ohio", "Wisconsin"
        "Iowa", "Kansas", "Minnesota", "Missouri", "Nebraska", "North Dakota","South Dakota"]
West = ["Arizona", "Colorado", "Idaho", "Montana", "Nevada", "New Mexico", "Utah","Wyoming"
        "Alaska", "California", "Hawaii", "Oregon","Washington"] 

# set median incomes by region(2014?)
nat_avgN = 53000
nat_avgW = 53000
nat_avgS = 46655
nat_avgM = 50000
df15_new["Median household income Value"][557] = "61,250" # only one missing data point
df15_new["Median household income Value"] = df15_new["Median household income Value"].str.replace(',','')
df15_new["Median household income Value"] = df15_new["Median household income Value"].apply(int) 

# Assign appropriate target variable. 1 if it is above median income, 0 otherwise
#for i in range(len(df15_new)):
#    county = df15.COUNTYCODE[i]
#    if str(county) == "0":
#        state = df15.County[i]
#        if state in South:
#            val = nat_avgS
#        elif state in North:           
#            val = nat_avgN
#        elif state in Mid:
#            val = nat_avgM
#        else:
#            val = nat_avgW
#        
#    if df15_new["Median household income Value"][i] >= val:
#        df15_new["Median household income Value"][i] = 1
#    else:
#        df15_new["Median household income Value"][i] = 0
        
# remove rows with 0 index for county
rm_list = []
for i in range(len(df15_new)):
    county = df15.COUNTYCODE[i]
    if str(county) == "0":
        rm_list.append(i)
df15_new = df15_new.drop(df15_new.index[rm_list])
df15_new = df15_new.reset_index(drop=True)

# remove nan data (simple cleaning)
df15_new = df15_new.dropna(thresh=16) # 16 is the number of columns in df15_new
for j in df15_new.columns:
    print(j, (df15_new[j].isnull().sum())/len(df15_new))
df15_new = df15_new.reset_index(drop=True) 
# Data is now at 1640 data points (down from 3191) 

# Check how many 1s and 0s the target variable has
print(df15_new["Median household income Value"].value_counts()) 

#assign the same median income check for all regions if want to look at things like poverty level income
df15_new["target"] = np.where(df15_new["Median household income Value"] > 45000, 1,0)
print(df15_new["target"].value_counts())

# lists the features used in this model
xcol = ["Adult smoking Value", "Adult obesity Value", "Physical inactivity Value", 
        "Excessive drinking Value","Alcohol-impaired driving deaths Value", 
        "Drug poisoning deaths Value","Poor mental health days Value", "Poor physical health days Value"]

X = df15_new[xcol]
y = df15_new["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4444)

# try grid search for logistic regression (does not really improve the results)
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }
clf = GridSearchCV(LogisticRegression(penalty='l2'), param_grid)
clf = clf.fit(X_train, y_train)
clf.best_params_
best_model = clf.best_estimator_
best_model.coef_
best_model.score(X_test,y_test)

#Logistic Regression
model_lr = LogisticRegression()
model_lr.fit(X_train, y_train)
pred_vals_lr = model_lr.predict(X_test)
score_lr = accuracy_score(y_test,pred_vals_lr)
print("logistic regression accuracy score: " + str(accuracy_score(y_test,pred_vals_lr)))
print("logistic regression precision score: " + str(precision_score(y_test,pred_vals_lr)))
print("logistic regression recall score: " + str(recall_score(y_test,pred_vals_lr)))
print("logistic regression f1 score: " + str(f1_score(y_test,pred_vals_lr)))
print(model_lr.coef_)
model_lr.fit(X_train/ np.std(X_train, 0),y_train)
print(model_lr.coef_)

X_logit = sm.add_constant(X)
t_log = sm.Logit(y,X_logit).fit()
print (t_log.summary())

#knn
maxscore = 0
numneigh = 0
knnscore_list = []
for i in range(1,21):
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X_train,y_train)
    y_pred = neigh.predict(X_test)
    score = accuracy_score(y_test,y_pred)
    knnscore_list.append(score)
    if score > maxscore:
        maxscore = score
        numneigh = i
print("k value with highest accuracy is " + str(numneigh) + " with score of " + str(maxscore))

model_gaus = GaussianNB()
model_gaus.fit(X_train, y_train)
pred_vals_gaus = model_gaus.predict(X_test)
score_gaus = accuracy_score(y_test,pred_vals_gaus)
print("Gaussian Naive Bayes score is " + str(score_gaus))

model_svc = SVC()
model_svc.fit(X_train, y_train)
pred_vals_svc = model_svc.predict(X_test)
score_svc = accuracy_score(y_test,pred_vals_svc)
print("SVM score is " + str(score_svc))

model_tree = DecisionTreeClassifier()
model_tree.fit(X_train, y_train)
pred_vals_tree = model_tree.predict(X_test)
score_tree = accuracy_score(y_test,pred_vals_tree)
print("Decision Tree score is " + str(score_tree))

model_rf = RandomForestClassifier()
model_rf.fit(X_train, y_train)
pred_vals_rf = model_rf.predict(X_test)
score_rf = accuracy_score(y_test,pred_vals_rf)
print("Random Forest score is " + str(score_rf))

models = {}
models['knn'] = KNeighborsClassifier(n_neighbors=4)
models['Logistic Regression'] = LogisticRegression()
models['Gaussian Naive Bayes'] = GaussianNB()
models['SVM'] = SVC()
models['Random Forest Classifier'] = RandomForestClassifier()
for name, model in models.items():  
    model.fit(X_train, y_train)
    y_pred_stats = model.predict(X_test)
    print("")
    print("Model: " + name) 
    print("accuracy score: " + str(accuracy_score(y_test,y_pred_stats)))
    print("precision score: " + str(precision_score(y_test,y_pred_stats)))
    print("recall score: " + str(recall_score(y_test,y_pred_stats)))
    print("f1 score: " + str(f1_score(y_test,y_pred_stats)))

# plot ROC 
models = {}
#models['knn'] = KNeighborsClassifier(n_neighbors=4)
models['Logistic Regression'] = LogisticRegression()
#models['Gaussian Naive Bayes'] = GaussianNB()
models['Random Forest Classifier'] = RandomForestClassifier()
for name, model in models.items():
    model.fit(X_train, y_train)
    pred_probs = model.predict_proba(X_test)     
    fpr, tpr, thresholds = roc_curve(y_test,pred_probs[:,1])
    plt.plot(fpr,tpr,label=name)
    print("")    
    print("Model: " + name) 
    print("AUC: " + str(roc_auc_score(y_test,pred_probs[:,1])))
    if name == "Random Forest Classifier":
        importances = model.feature_importances_    
plt.legend(loc="best")
plt.show()
print(importances)

# try SGD classifier
clf_SGD = linear_model.SGDClassifier(n_iter=30)
clf_SGD.fit(X_train, y_train)
accuracy_score(y_test, clf_SGD.predict(X_test))
print("SGD accuracy score is " + str(accuracy_score(y_test, clf_SGD.predict(X_test))))

# explore the data, check the distribution of each important feature
plt.hist(X["Adult obesity Value"], bins = 10)
plt.show()
plt.hist(X["Physical inactivity Value"],bins = 10)
plt.show()
plt.hist(X["Adult smoking Value"],bins = 10)
plt.show()
plt.hist(df15_new["Excessive drinking Value"],bins = 10)
plt.show()
plt.hist(df15_new["Median household income Value"],bins = 50)
plt.show()
plt.hist(df15_new["Poor mental health days Value"],bins = 50)
plt.show()

plt.plot(X["Adult obesity Value"], df15_new["Median household income Value"],'.')
plt.show()
plt.plot(X["Adult smoking Value"], df15_new["Median household income Value"],'.')
plt.show()
plt.plot(X["Excessive drinking Value"], df15_new["Median household income Value"],'.')
plt.show()

# write to csv to do d3 vis
# the name of columns of the csv file will need to change to match current html file for d3
X_test['target'] = y_test
pred_vals_lr
templist = []
badvals = []
countbad = 0
countFP = 0
countFN = 0
FP_list = [0]*len(y_test)
FN_list = [0]*len(y_test)
ilistFP = []
ilistFN = []
for i in range(len(X_test)):
    if X_test["target"].iloc[i] == 1:
        templist.append("Above Median Income")
    if X_test["target"].iloc[i] == 0:
        templist.append("Below Median Income")
    if pred_vals_lr[i] != X_test["target"].iloc[i]:
        badvals.append(1) 
        countbad+=1
        if pred_vals_lr[i] == 1:
            FP_list[i] = 1
            countFP += 1
            ilistFP.append(i)
        if pred_vals_lr[i] == 0:
            FN_list[i] = 1
            countFN += 1
            ilistFN.append(i)
    if pred_vals_lr[i] == X_test["target"].iloc[i]:
        badvals.append(0)
X_test["target2"] = templist
X_test["badval"] = badvals
X_test["FN"] = FN_list
X_test["FP"] = FP_list
X_test.to_csv("output.csv",sep = ',')

# create averages of important features for d3 spider graph
# since all these features are about normally distributed, will use the mean
smoke_below = 0
smoke_above = 0
drink_below = 0
drink_above = 0
obese_below = 0
obese_above = 0
inactive_below = 0
inactive_above = 0
drugs_above = 0
drugs_below = 0
mental_below = 0
mental_above = 0
drunkdrive_above = 0
drunkdrive_below = 0
above = 0
below = 0
physical_above = 0
physical_below = 0
for i in range(len(X)):
    if y[i] == 1:
        smoke_above+=X["Adult smoking Value"][i]
        drink_above+=X["Excessive drinking Value"][i]
        obese_above+=X["Adult obesity Value"][i] 
        inactive_above+=X["Physical inactivity Value"][i]
        drugs_above+=X["Drug poisoning deaths Value"][i]
        mental_above+=X["Poor mental health days Value"][i]
        drunkdrive_above=X["Alcohol-impaired driving deaths Value"][i]
        physical_above=X["Poor physical health days Value"][i]
        above += 1
    if y[i] == 0:
        smoke_below+=X["Adult smoking Value"][i]
        drink_below+=X["Excessive drinking Value"][i]
        obese_below+=X["Adult obesity Value"][i] 
        inactive_below+=X["Physical inactivity Value"][i]
        drugs_below+=X["Drug poisoning deaths Value"][i]
        mental_below+=X["Poor mental health days Value"][i]
        drunkdrive_below=X["Alcohol-impaired driving deaths Value"][i]
        physical_below=X["Poor physical health days Value"][i]
        below += 1

smoke_below = smoke_below/below
smoke_above = smoke_above/above
drink_below = drink_below/below
drink_above = drink_above/above
obese_below = obese_below/below
obese_above = obese_above/above
inactive_below = inactive_below/below
inactive_above = inactive_above/above
drugs_above = drugs_above/above
drugs_below = drugs_below/below
mental_below = mental_below/below
mental_above = mental_above/above  
drunkdrive_above = drunkdrive_above/above
drunkdrive_below = drunkdrive_below/below 
physical_above = physical_above/above
physical_below = physical_below/below


