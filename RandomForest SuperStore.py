# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 18:03:30 2020

@author: Calvin
"""



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os



Datafolder = r'C:\Users\Calvin\Documents\KPMG\Calvin created'

#WANA - Sheet1
GlobalSuperStore = pd.read_excel(os.path.join(Datafolder,'Global Superstore 2018.xlsx'))

missing_values = GlobalSuperStore.isna().sum()
print(missing_values)

GlobalSuperStore = GlobalSuperStore.dropna( how = 'any', thresh = 40000, axis = 1)


import seaborn as sns
corr = GlobalSuperStore.corr()
sns.heatmap(corr, annot = True)

#General Plotting
for col in GlobalSuperStore.columns:
#    GlobalSuperStore[col].plot()
    print(GlobalSuperStore[col].describe())
    
    
GlobalSuperStore1 = GlobalSuperStore
GlobalSuperStore1 = GlobalSuperStore1.drop('Row ID', axis = 1)
#GlobalSuperStore = GlobalSuperStore.drop('Order ID', axis = 1)
GlobalSuperStore1 = GlobalSuperStore1.drop('Ship Date', axis = 1)
#GlobalSuperStore1 = GlobalSuperStore1.drop('Customer Name', axis = 1)
GlobalSuperStore1 = GlobalSuperStore1.drop('Discount', axis = 1)
GlobalSuperStore1 = GlobalSuperStore1.drop('Product ID', axis = 1)
GlobalSuperStore1 = GlobalSuperStore1.drop('Product Name', axis = 1)
GlobalSuperStore1 = GlobalSuperStore1.drop('Shipping Cost', axis = 1)

#GlobalSuperStore1 =  GlobalSuperStore.drop('Shipping Cost', axis = 1)

Shipmode = pd.get_dummies(GlobalSuperStore1['Ship Mode']) #good
Segment = pd.get_dummies(GlobalSuperStore1['Segment']) #good
SubCategory  = pd.get_dummies(GlobalSuperStore1[ 'Sub-Category']) #turn into numerical data
country  = pd.get_dummies(GlobalSuperStore1['Country']) # turn into numerical data




GlobalSuperStore1 = pd.concat([GlobalSuperStore1[['Sales']],Shipmode,Segment, SubCategory, country] , axis = 1 )


labels = np.array(GlobalSuperStore1['Sales'])
features = GlobalSuperStore1.drop('Sales', axis=1)

feature_list = list(features.columns)
features = np.array(features)


#ShipMode1 = pd.get_dummies(GlobalSuperStore['Ship Mode'])

#AGGGlobalSuperStore = GlobalSuperStore.groupby(['Product Id'], )





from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, labels, 
                                                    train_size=.8, shuffle=False, random_state=1)

from sklearn.ensemble import RandomForestRegressor
random_forest = RandomForestRegressor()
random_forest.fit(X_train, y_train)
print(random_forest.score(X_train, y_train))


train_predictions = random_forest.predict(X_train)
test_predictions = random_forest.predict(X_test)
plt.scatter(train_predictions, y_train , label='train')
plt.scatter(test_predictions, y_test, label='test')
plt.legend()
plt.show()


# feature importances from random forest model
importances = random_forest.feature_importances_
# index of greatest to least feature importances


print(importances)
sorted_index = np.argsort(importances)[::-1] #start, stop, and step.  ::-1 reverses the list
sorted_index =sorted_index[:30]
x = range(len(sorted_index))


# create tick labels
labels = np.array(feature_list)[sorted_index]
plt.bar(x, importances[sorted_index], tick_label=labels)
# rotate tick labels to vertical
plt.xticks(rotation=90)
plt.show()

