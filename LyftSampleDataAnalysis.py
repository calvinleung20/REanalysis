# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 17:04:36 2019

@author: Calvin
"""
# sample data for cab
# knowing the data is the first step
# so what we need to do is to understand how the data works 

# high level plan
# 1) understand the problem - look at each variable and quantitative analysis on the impact and how it will play as a factor
# 2) univariable study - focus on the dependent variable (what are you looking for at the end ? any trends?)
# 3) multivariate study - focus on how the depedent vairable and independaent variables relate
# 4) basic cleaning - handle missing data and categorical variables
# 5) test assumptions - test to see if the data meets the assumptions required by most multivarite techniques

#packages as usual
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
import plotly as py

import plotly.graph_objects as go

# f = go.FigureWidget()
# f


#importing  data 

df_taxi = pd.read_csv(r'C:\Users\Calvin\Documents\Lyft Case\cab_june_2018_sample.csv')
#Index(['cab_type', 'pickup_datetime', 'dropoff_datetime', 'trip_distance',
#       'pulocationid', 'dolocationid', 'ratecodeid', 'payment_type',
#       'total_fare')

df_taxi_lookup_sample = pd.read_csv(r'C:\Users\Calvin\Documents\Lyft Case\taxi_lookup_sample.csv')
#Index(['lookup_type', 'lookup_id', 'lookup_value'], dtype='object')
df_taxi_zone_lookup_sample = pd.read_csv(r'C:\Users\Calvin\Documents\Lyft Case\taxi_zone_lookup_sample.csv')
#Index(['locationid', 'borough', 'zone', 'service_zone'], dtype='object')
print(df_train.head(10))

df_taxifull = df_taxi.merge(df_taxi_lookup_sample, left_on='lkey', right_on='rkey')


df_taxi.columns
df_taxi_lookup_sample.columns
df_taxi_zone_lookup_sample.columns



# Here you can see the top ten rows, primary key on the left, there are columns like alley, miscfeature that have missing values.
# it seems that the categorical values are the ones that are missing the most 
# dive deeper with description

# qualitative column description:
#     mssubclass - no idea
#     mszoning - not sure, possible values rl and RM


#Analysis of cab type
df_taxi['cab_type'].value_counts()

#lets check if its useable

#putting it into a SCATTER PLOT TO SEE IF THERE IS ANY CORRELATION

import plotly.graph_objects as go

trace = go.Scatter(x=df_train['trip_distance'],
                   y=df_train['total_fare'],
                   mode='markers')
data=[trace]
layout = go.Layout(title='Trip_distance vs Saleprice Scatter')

figure = go.Figure(data=data, layout=layout)
figure.write_html('distance vs fare.html', auto_open=True)


## AFTER OUTPUT
# Very interesting... output reminds me of a boxplot... NO STRONG CORRELATION... 
# The categorical values make it essentially irelevant in a scatterplot, i could order the outputs but i might as well find the 
# stats via a boxplot.



import plotly.graph_objects as go

trace = go.Box(x=df_train['Neighborhood'],
                   y=df_train['SalePrice'], notched=True
                 )
data=[trace]
layout = go.Layout(title='Neighborhood vs Saleprice Boxplotter')

figure = go.Figure(data=data, layout=layout)

f2 = go.FigureWidget(figure)
f2

#oldtown chicago.... thats where old town road is from!!!
#all these places are in illinois
# looking at noridge on google doesnt make it


# NOTCH DESCRIBES THE CONDIENCE INTERVAL (BY DEFAULT 95 PERCENT)
# condience interval is an interval estimate combined with a probability statement 



##THOUGHTS ON THE BOXPLOT
#boxplot is a standardized way of displaying the distribution of data based on a five number summary
# minimum(Q1-1.5*IQR), Q1, median, Q3, maximum(Q3+1.5*IQR)
#IQR= Q3(75th percentile)-Q1(25th Percentile) OR 50 PERCENT OF THE DATA
# describes shape of data (or groups of categorical data)
# it can tell you about your outliers and their values
# data symetrical ? how tightly grouped ? and if the data is skewed

#percentages from PDF - specify probability of the random variable falling within a particular range of values 


# z-critical value:
# 1.959963984540054
# Confidence interval:
# (328396.7497315812, 343446.94226841885)



#side note - graphing a PDF for normal distribution
x = np.linspace(-4, 4, num = 100)
constant = 1.0 / np.sqrt(2*np.pi)
pdf_normal_distribution = constant * np.exp((-x**2) / 2.0)
fig, ax = plt.subplots(figsize=(10, 5));
ax.plot(x, pdf_normal_distribution);
ax.set_ylim(0);
ax.set_title('Normal Distribution', size = 20);
ax.set_ylabel('Probability Density', size = 20);

#this graph does not show the probability, it shows the probability density 
#so to get the probability we will need to integrate for a range .... remeber 68-95-99.7 Where your data needs to be for normal dist


#example: What is the probability of a random data point landing within the interquartile range .6745 standard deviation 
# from the mean ? 
# integrate from -.675,.675 

# Make PDF for the normal distribution(simplified mean =0 and standard dev = 1) a function 
def normalProbabilityDensity(x):
    constant = 1.0 / np.sqrt(2*np.pi)
    return(constant * np.exp((-x**2) / 2.0) )

from scipy.integrate import quad
# Integrate PDF from -.6745 to .6745
result, _ = quad(normalProbabilityDensity, -.6745, .6745, limit = 1000)
print(result)
print(_)
# therefore 50% of the values are within .6745stddev

import math
df_taxi[['cab_type','trip_distance']].boxplot()


import math
df_taxi[['cab_type','trip_distance']].boxplot()

#sample 

sample_size = 1000
sample = np.random.choice(a= df_train['SalePrice'][df_train['Neighborhood']=='NoRidge'], size = sample_size)
sample_mean = sample.mean()

z_critical = stats.norm.ppf(q = 0.975)  # Get the z-critical value*

print("z-critical value:")              # Check the z-critical value
print(z_critical)                        

pop_stdev = df_taxi['trip_distance'][df_train['Neighborhood']=='NoRidge'].std()  # Get the population standard deviation

margin_of_error = z_critical * (pop_stdev/math.sqrt(sample_size))

confidence_interval = (sample_mean - margin_of_error,
                       sample_mean + margin_of_error)  

print("Confidence interval:")
print(confidence_interval)



#sample 

sample_size =df_train['SalePrice'][df_train['Neighborhood']=='NoRidge'].count()
sample = np.random.choice(a= df_train['SalePrice'][df_train['Neighborhood']=='NoRidge'], size = sample_size)
sample_mean = sample.mean()

z_critical = stats.norm.ppf(q = 0.975)  # Get the z-critical value*

print("z-critical value:")              # Check the z-critical value
print(z_critical)                        

pop_stdev = df_train['SalePrice'][df_train['Neighborhood']=='NoRidge'].std()  # Get the population standard deviation

margin_of_error = z_critical * (pop_stdev/math.sqrt(sample_size))

confidence_interval = (sample_mean - margin_of_error,
                       sample_mean + margin_of_error)  

print("Confidence interval:")
print(confidence_interval)

print(df_train['SalePrice'][df_train['Neighborhood']=='NoRidge'])
sns.distplot(df_train['total_fare']);
#deviate from normal
#appreciable positive skewness
#show peakness

# skenesss is a measure of asymmetrey of the probability distribution 
# skewed to the left - means the tail on the left hand side is longer
# skweed to the right  - means the tail on the righ thand side is longer

# Kurtosis - existence of outliers
# measure of whether data is heavy tailed (lots of outliers) or light tailed (lack of outliers) relative to normal dist
#skewness and kurtosis
print("Skewness: %f" % df_train['trip_distance'].skew())
print("Kurtosis: %f" % df_train['trip_distance'].kurt())
#correlation matrix


corrmatdf = df_taxi.dropna()
corrmat = corrmatdf.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);

trace = go.Heatmap(
        z=corrmat,
        x=corrmat.columns[::-1],
        y=corrmat.columns[::-1],
        colorscale='Viridis')

data=[trace]
layout = go.Layout(title='OverallCond vs Saleprice Scatter')

figure = go.Figure(data=data, layout=layout)
figure.update_layout(
    title='Correlation matrix (heatmap style)',
    xaxis_nticks=36)
f2 = go.FigureWidget(figure)
f2

#saleprice correlation matrix
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));


sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df_train[cols], size = 2.5)
plt.show();