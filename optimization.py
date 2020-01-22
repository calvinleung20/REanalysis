# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 22:00:16 2019

@author: Calvin
"""


import numpy as np
import pandas as pd


#specialists
average_laboursneeded = 100
SD = 30
InHouseCostperhour = 150
ExternalConsultancyCosthr = 450



mu, sigma = average_laboursneeded, SD # mean and standard deviation





datalist = []
for Inhousestaff in range(60,150,5):
#    s = np.random.normal(mu, sigma, 1000)
    for i in range(10000):
        Staff_Required = np.random.normal(mu, sigma, 1)
        Staff_Required = Staff_Required[0]
        ExternalStaff_Required = max([0,Staff_Required-Inhousestaff])
        
        
        ## costs
        InHouseCost = InHouseCostperhour*Inhousestaff
        ExternalCost = ExternalConsultancyCosthr*ExternalStaff_Required
        Totalcost = InHouseCost+ExternalCost
        datalist.append([Inhousestaff,Staff_Required,ExternalStaff_Required,InHouseCost,ExternalCost,Totalcost])
        
     
data = pd.DataFrame(data = datalist , columns = ['Inhousestaff','Staff_Required','ExternalStaff_Required','InHouseCost','ExternalCost','Totalcost'] )

meandata = data.groupby(['Inhousestaff']).mean()
#meandatagraph = meandata['']
lines = meandata.plot.line( y = 'Totalcost')
plt.plot()
plt.show()

#density1 = 
data['Totalcost'].plot.hist(bins = 30, density = True)


#density = data.plot.hist(column = 'Totalcost' , by = 'Totalcost' ,bins = 30, density = True)
#density1 = data.plot.hist(by = 'Totalcost',bins = 30, density = False)

#data.plot.hist(bins = 30, density = True)


import matplotlib.pyplot as plt
#
#import matplotlib.pyplot as plt
#count, bins, ignored = plt.hist(s, 30, density=True) 
#plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
#np.exp( - (bins - mu)**2 / (2 * sigma**2) ), linewidth=2, color='r')
#plt.show()
#

#x = np.linspace(-4, 4, num = 100)
#constant = 1.0 / np.sqrt(2*np.pi)
#pdf_normal_distribution = constant * np.exp((-x**2) / 2.0)
#fig, ax = plt.subplots(figsize=(10, 5));
#ax.plot(x, pdf_normal_distribution);
#ax.set_ylim(0);
#ax.set_title('Normal Distribution', size = 20);
#ax.set_ylabel('Probability Density', size = 20);


plt.plot()
plt.show()



