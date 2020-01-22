# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 15:29:22 2019

@author: Calvin
"""

# -*- coding: utf-8 -*-

"""

Created on Wed Feb  6 16:50:17 2019

 

@author: CaLeung

"""

 

 

file1 =  r"C:\\Users\\Calvin\\Documents\\Real Estate\\Output\\outputdata.csv"

 

import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
 

data = pd.read_csv(file1)

print(data.head())

 

data['Last Update:'] = pd.to_datetime(data['Last Update:'])

data = data.drop((data[data['Last Update:'] < '2017-01-01']).index)

 

data['Original:'] = data['Original:'].apply(lambda x: x.replace('$',''))

data['Original:'] = data['Original:'].apply(lambda x: pd.to_numeric(x.replace(',',''), errors='coerce'))

#filteredMLS3['Contract Date:'] = pd.to_datetime(filteredMLS3['Contract Date:'])

 

 

 

fig, ax = plt.subplots(figsize=(12, 6))

data.plot('Last Update:', 'Original:', ax=ax)

ax.set(title="Original Price vs Time")

plt.show()


#rolling
Original_price_smooth = data['Original:'].rolling(30).mean()
Original_price_smooth.plot(figsize=(10, 5))
plt.title("Rolled mean price")
plt.show()
#means = np.mean(Original_price_smooth, axis=0)
#
#maxs = np.max(Original_price_smooth, axis=-1)
#
#stds = np.std(Original_price_smooth, axis=-1)
#
#fig, ax = plt.subplots(figsize=(12, 6))
#
#data.plot('Last Update:', 'Original:', ax=ax)
#
#ax.set(title="Original Price vs Time")
#
#plt.show()
#



 ### model for regression DOM and Original price

X=data[['DOM:']]
y= data['Original:'].values.T

from sklearn import linear_model
model = linear_model.LinearRegression()
model.fit(X,y)
new_inputs = np.array(np.random.randint(0,60, size = 100))

predictions = model.predict(new_inputs.reshape([-1,1]))

plt.scatter(X['DOM:'], y)
plt.xlabel('inputs')
plt.ylabel('predictions')
plt.show()


# Visualize the inputs and predicted values
plt.scatter(new_inputs, predictions, color='r', s=3)
plt.xlabel('inputs')
plt.ylabel('predictions')
plt.show()

### ## Classifier


X=data[['DOM:','Original:']]

 

y= data['No_Val_Available_8'].values.T

#y= data['No_Val_Available_8'].values.reshape([1,-1]).T

 

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X, y, train_size = .8, shuffle=True, random_state =1)

 

plt.scatter( X_train['DOM:'], X_train['Original:'] , c=pd.factorize(y_train)[0], cmap= plt.cm.coolwarm   )

plt.title("Test data community values vs price")

plt.show()


 

 

# Import a support vector classifier

from sklearn.svm import LinearSVC

# Instantiate this model

model = LinearSVC()

# Fit the model on some data

model.fit(X_train, y_train)
 

predictions = model.predict(X_test)

print(predictions)

plt.scatter( X_test['DOM:'], X_test['Original:'], c=pd.factorize(predictions)[0], cmap= plt.cm.coolwarm   )

plt.title("Predicted community values ")

plt.show()

from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=3)

print(scores)


#
# 
#

#
# 

### approx 

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
 
prices = data[['Last Update:','Original:']].set_index('Last Update:')

prices.plot(legend=False)
plt.tight_layout()
plt.show()





 

###### Time series

 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
 
prices = data[['Last Update:','Original:']].set_index('Last Update:')
preddates = pd.DataFrame( pd.date_range(start='2017-07-27', end='2019-12-21'))
#preddates.merge(prices, left_on= index, right_on='Last Update:')
ns = prices index=dr)
(np.random.randn(3),
prices.plot(legend=False)
plt.tight_layout()
plt.show()



X=data[['Last Update:']]

y=data[['Original:']]



# Count the missing values of each time series
missing_values = prices.isna().sum()
print(missing_values)

#scores = cross_val_score(Ridge(), X, y, cv=3)
#print(scores)



# Split our data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    train_size=.8, shuffle=False, random_state=1)

# Fit our model and generate predictions
model = Ridge()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
score = r2_score(y_test, predictions)
print(score)

 