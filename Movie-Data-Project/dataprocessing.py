# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 11:06:40 2016

@author: avasilye
"""

import pandas as pd
import datetime as dt
import numpy as np
import glob
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn
from sklearn.linear_model import LinearRegression
from sklearn import cross_validation, datasets, linear_model, grid_search,tree
from scipy import stats

list_ = []
for filename in glob.glob('*.csv'):
    df = pd.read_csv(filename,index_col = None, header = 0,skip_blank_lines=True)
    list_.append(df)
frame = pd.concat(list_,ignore_index=True)

df = frame.drop(frame.columns[[6,7,8]],axis=1)

# clean out all the "nan" in the data
count1 = 0
theater_list = df.ix[:,1]
for i in range(len(theater_list)):
    theater_list[i] = str(theater_list[i]).lstrip()
    theater_list[i] = theater_list[i].split(' ')[0]
    if theater_list[i] == 'nan':
        count1 += 1
theater_list = [i.rstrip(',') for i in theater_list]

count2 = 0
reldays_list = df.ix[:,2]
for i in range(len(reldays_list)):
    reldays_list[i] = str(reldays_list[i]).lstrip()
    reldays_list[i] = reldays_list[i].split(' ',1)[0]
    if reldays_list[i] == 'nan':
        count2 += 1
reldays_list = [i.rstrip(',') for i in reldays_list]

count3 = 0
domestic_list = df.ix[:,3]
for i in range (len(domestic_list)):
    domestic_list[i]=str(domestic_list[i]).strip('b\'')
    domestic_list[i]=str(domestic_list[i]).strip('$')
    temp = domestic_list[i].split(',')
    domestic_list[i]=''.join(temp)
    if domestic_list[i] == 'nan':
        count3 += 1
        
count4 = 0
budget_list = df.ix[:,6]
for i in range (len(budget_list)):
    budget_list[i]=str(budget_list[i]).strip('b\'')
    budget_list[i]=str(budget_list[i]).strip('$')
    temp = budget_list[i].split(',')
    budget_list[i]=''.join(temp)
    if budget_list[i] == 'N/A':
        count4 += 1
        
count5 = 0
world_list = df.ix[:,7]
for i in range (len(world_list)):
    world_list[i]=str(world_list[i]).strip('b\'')
    world_list[i]=str(world_list[i]).strip('$')
    temp = world_list[i].split(',')
    world_list[i]=''.join(temp)
    if world_list[i] == 'nan':
        count5 += 1

count6 = 0
title_list = df.ix[:,0]
for i in range (len(title_list)):
    title_list[i]=str(title_list[i]).strip('b\'')
    if title_list[i] == 'nan':
        count6 += 1

count7 = 0
dates_list = df.ix[:,5]
for i in range(len(dates_list)):
    if type(dates_list[i])==str and i != 17 and dates_list[i] != 'TBD' and i != 773:
        dates_list[i] = (dt.datetime.strptime(dates_list[i], '%B %d, %Y')) 
    else:
        dates_list[i] = 'nan'
        count7 += 1


df = df[df["theaters"].str.contains("nan") == False]
df = df[df["release"].str.contains("nan") == False]
df = df.reset_index(drop=True)

df_dist = pd.read_csv('additional_data/distributor_table.csv',index_col = None, header = 0,skip_blank_lines=True)
titDist_list = df_dist.ix[:,1]
rankDist_list = df_dist.ix[:,0]
shareDist_list = df_dist.ix[:,5]

# a very messy way to merge the distributor share into the current dataframe
dist_list = df.ix[:,4]
rank_list= ["nan"]*len(dist_list)
share_list = ["nan"]*len(dist_list)
distNew_list = ["nan"]*len(dist_list) 
for i in range(len(dist_list)):
    if dist_list[i] == "Sony Classics":
        dist_list[i] = "Sony Pictures Classics"
    if dist_list[i] == "Weinstein Company":
        dist_list[i] = "Weinstein Co"
    if dist_list[i] == "AMC Theaters":
        dist_list[i] = "AMC Independent"
    if dist_list[i] == "Palisades Tartan":
        dist_list[i] = "Palisades"
    if dist_list[i] == "Picture This!":
        dist_list[i] = "Picture This"  
    if dist_list[i] == "City Lights Pictures Releasing":
        dist_list[i] = "City Lights"  
    if dist_list[i] == "AdLabs":
        dist_list[i] = "Adlab"
    if dist_list[i] == "Paramount Classics":
        dist_list[i] = "Paramount Pictures"
    if dist_list[i] == "Lions Gate":
        dist_list[i] = "Lionsgate"
    if dist_list[i] == "Rainbow Films":
        dist_list[i] = "Rainbow"
    if dist_list[i] == "Vid.":
        dist_list[i] = "Video Sound"
    if dist_list[i] == "Artmattan Prods.":
        dist_list[i] = "Artmattan"
    if dist_list[i] == "Zee TV":
        dist_list[i] = "Zee"
    if dist_list[i] == "Weinstein / Dragon Dynasty":
        dist_list[i] = "Weinstein Co."
    if dist_list[i] == "Maya Entertainment":
        dist_list[i] = "Maya"
    if dist_list[i] == "Laemmle / Zeller Films":
        dist_list[i] = "Laemmle"
    if dist_list[i] == "Arab Film Dist.":
        dist_list[i] = "Arab"
    if dist_list[i] == "Sundance Film Series":
        dist_list[i] = "Sundance"
    if dist_list[i] == "Indomina Media":
        dist_list[i] = "Indomina"
    if dist_list[i] == "Viz Media":
        dist_list[i] = "Viz"
    for j in range(len(titDist_list)):
        if dist_list[i].lower() in titDist_list[j].lower() or titDist_list[j].lower() in dist_list[i].lower():
            rank_list[i] = str(rankDist_list[j])
            share_list[i] = str(shareDist_list[j])
            distNew_list[i] = str(titDist_list[j])
            break
        
df['dist new'] = distNew_list 
df['dist rank'] = rank_list
df['share'] = share_list

df = df[df["share"].str.contains("nan") == False]
df = df.reset_index(drop=True)

time_list = df.ix[:,5]
tbound_str = "December 31, 1994"
tbound = dt.datetime.strptime(tbound_str,'%B %d, %Y')
tind_list = []
for i in range(len(time_list)):
    if time_list[i] < tbound:
        tind_list.append(i)

df = df.drop(df.index[tind_list])
df = df.reset_index(drop=True)

# get oscars into the dataframe
df_oscar = pd.read_csv('additional_data/oscars.csv',index_col = None, header = 0,skip_blank_lines=True)
oscar_list = df_oscar.ix[:,1]
title_list = df.ix[:,0]
oscarYes_list = [0]*len(title_list) 
counter_oscar = 0
for i in range(len(title_list)):
     for j in range(len(oscar_list)):
        if str(title_list[i]).lower() == str(oscar_list[j]).lower(): #or str(oscar_list[j]).lower() in str(title_list[i]).lower():
            oscarYes_list[i] = 1
            counter_oscar+=1
            break
oscarYes_list[58]=1 # Amelie
oscarYes_list[938]=1 # Pan's Labyrinth

df['oscar'] = oscarYes_list
print(df.iloc[728][0])
df = df.drop(df.index[728]) # drops Life is Beautiful dubbed
df = df.reset_index(drop=True)

# add seasons to the dataframe
testdate= df.ix[:,5]
spring_list = [0]*len(testdate)
summer_list = [0]*len(testdate)
fall_list = [0]*len(testdate)
winter_list = [0]*len(testdate)
for i in range(len(testdate)):
    if testdate[i].month == 3 or testdate[i].month == 4 or testdate[i].month == 5:
        spring_list[i] = 1
    if testdate[i].month == 6 or testdate[i].month == 7 or testdate[i].month == 8:
        summer_list[i] = 1
    if testdate[i].month == 9 or testdate[i].month == 10 or testdate[i].month == 11:
        fall_list[i] = 1
    if testdate[i].month == 12 or testdate[i].month == 1 or testdate[i].month == 2:
        winter_list[i] = 1

df['spring'] = spring_list
df['summer'] = summer_list        
df['fall'] = fall_list    

df = df.drop(df.index[[58,585,937]]) # drops top three domestic gross outliers (calculated elsewhere)
df = df.reset_index(drop=True)

df = df.drop(df.index[[830,654,981]]) # drops top three domestic gross outliers (calculated elsewhere)
df = df.reset_index(drop=True)

df = df.drop(df.index[[526,264,541]]) # drops top three domestic gross outliers (calculated elsewhere)
df = df.reset_index(drop=True)

df = df.drop(df.index[[999,1000,484,908,1302]]) # drops top three domestic gross outliers (calculated elsewhere)
df = df.reset_index(drop=True)

Y = df.ix[:,3]
Y = [i.replace(',','') for i in Y]

for i in range(len(Y)):
    Y[i] = float(Y[i])
    #Y[i] = np.log(Y[i])
    
plt.hist(Y)    
plt.rcParams['xtick.labelsize'] = 10 
plt.rcParams['ytick.labelsize'] = 12
plt.autoscale(enable=True, axis='y', tight=True)
x = [0, 2000000, 4000000, 6000000,8000000,10000000,12000000,14000000,16000000]
#plt.xticks(x, labels, rotation='vertical')
plt.xticks(x,['0','2M', '4M', '6M', '8M','10M','12M','14M','16M'])
ax = plt.gca()
plt.xlabel('Domestic Gross',fontsize=18)
plt.ylabel("Number of Movies",fontsize=18)
plt.show()

X = df.drop(df.columns[[0,3,4,5,6,7,8,9]],axis=1) # so only keep theaters, release, share, oscars, and seasons
X = X.replace(',','')

temp1 = X.ix[:,0]
temp1 = [i.replace(',','') for i in temp1]
for i in range(len(temp1)):
    temp1[i]= float(temp1[i])
    
temp2 = X.ix[:,1]
temp2 = [i.replace(',','') for i in temp2]
for i in range(len(temp2)):
    temp2[i]= float(temp2[i])
    
temp3 = X.ix[:,2]
temp3 = [i.replace(',','') for i in temp3]
for i in range(len(temp3)):
    temp3[i]= float(temp3[i])

X['theaters'] = temp1 
X['release'] = temp2
X['share'] = temp3

plt.scatter(X.theaters,Y, alpha=.3)
plt.rcParams['xtick.labelsize'] = 16 
plt.rcParams['ytick.labelsize'] = 12
plt.autoscale(enable=True, axis='y', tight=True)
y = [0, 2000000, 4000000, 6000000,8000000,10000000,12000000,14000000,16000000]
plt.yticks(y,['0','2M', '4M', '6M', '8M','10M','12M','14M','16M'])
ax = plt.gca()
plt.xlabel('Number of Theaters',fontsize=18)
plt.ylabel("Domestic Gross",fontsize=18)
plt.show()

#Y = np.log(Y) # to have Y take a more normal distribution

# get only movies over 2M
#Y_indList = []
#Y_new=[]
#for i in range(len(Y)):
#    if Y[i] < 2000000:
#        Y_indList.append(i)
#    if Y[i] >= 2000000:
#        Y_new.append(Y[i])
#        

def scatter_matrix(X):
    feature_count = len(X.columns)
    fig,ax = plt.subplots(ncols=feature_count,nrows=feature_count,figsize=(5*feature_count,5*feature_count))

    for i,feature_i in enumerate(X):
        for j,feature_j in enumerate(X):
            ax[i][j].scatter(X[feature_i],X[feature_j])
            ax[i][j].set_xlabel('Feature ' + str(feature_j))
            ax[i][j].set_ylabel('Feature ' + str(feature_i))

scatter_matrix(X)


X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=0.3) #,random_state=5)

X = sm.add_constant(X)
model = sm.OLS(Y,X)
results = model.fit()
results.summary()

#
#plt.hist(stats.zscore(results.resid), bins=50);
#from statsmodels import graphics
#graphics.gofplots.qqplot(results.resid, line='r') shows how far youre off noramlly distributed data
#

models = {}
models['lin_reg'] = linear_model.LinearRegression()
models['ridge'] = linear_model.Ridge(tol = 0.001)
models['lasso'] = linear_model.Lasso(alpha=.2, tol = 0.001)
#models['lasso'] = linear_model.Lasso(alpha=0.035906445401860804, tol = 0.001)
models['elasticnet'] = linear_model.ElasticNet(tol = 0.001)  

#lasso = linear_model.Lasso()
#parameters = {'normalize':(True,False),
#              'alpha':np.logspace(-4,-.1,30)}
#grid_searcher = grid_search.GridSearchCV(lasso, parameters)
#grid_searcher.fit(X, Y) 
#grid_searcher.best_params_
#best_model = grid_searcher.best_estimator_
#best_model.coef_
#best_model.score(X_test,y_test)
#

for name,model in models.items():
    model.fit(X_train,y_train)
    print('Model: '+name)
    print("Score: " + str(model.score(X_test,y_test)))
    sorted_features = sorted(zip(X,model.coef_), key=lambda tup: abs(tup[1]), reverse=True)
    for feature in sorted_features:
        print(feature)
        
    print("") 


# plot predicted vs actual Y using indices of sorted actual Y
X["target"] = Y
# model data taken from OLS summary as this gives the best R-square
#X["model"] = 0.1463*X.oscar + 0.1677*X.fall+ 0.2055*X.summer + 0.1544*X.spring + 0.0466*X.share+ 0.0275*X.theaters + 0.0082*X.release+9.9207
X["model"] = 6.553e+05*X.oscar + 4.883e+04*X.fall+ 9.869e+04*X.summer + 1.137e+04*X.spring + 1.585e+04*X.share+ 2.129e+04 *X.theaters + 4500.9035*X.release-4.734e+05
sort_test = X.sort_values(['target'])
o = sort_test["target"]
oo = sort_test["model"]
y = [-2000000, 2000000, 4000000, 6000000,8000000,10000000,12000000,14000000,16000000]
plt.yticks(y,['-2M','0','2M', '4M', '6M', '8M','10M','12M','14M','16M'])
ax = plt.gca()
plt.plot(list(oo),'r.',label = "predicted")
plt.plot(list(o),'.',label = "actual")
legend = plt.legend(loc= "best", numpoints = 1, shadow = True)
plt.setp(plt.gca().get_legend().get_texts(), fontsize='12') 
plt.xlabel('Sorted Index',fontsize=18)
plt.ylabel("Domestic Gross",fontsize=18)
plt.show()
