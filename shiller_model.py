# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 09:54:38 2019

@author: jcondori
"""

''' Importacion Data'''

import quandl
import matplotlib.pyplot as plt
import numpy as np
#mydata = quandl.get("FRED/GDP")
#mydata=quandl.get()
#mydata.plot()

''' Extrayendo'''

#mydata=quandl.get(["MULTPL/SHILLER_PE_RATIO_MONTH"\
#                    ,"MULTPL/SP500_REAL_PRICE_MONTH",\
#                    "MULTPL/SP500_DIV_YIELD_MONTH"],\
#                     authtoken="jXzqHEmopssr9P9jayeC")
df=mydata.copy()

df=df.reset_index()

df['YearMonth'] = df['Date'].map(lambda x: 100*x.year + x.month)

df=df.groupby(['YearMonth']).sum()

df=df[['MULTPL/SHILLER_PE_RATIO_MONTH - Value',
       'MULTPL/SP500_REAL_PRICE_MONTH - Value',
       'MULTPL/SP500_DIV_YIELD_MONTH - Value']].replace({0:np.nan})

# llenar NA con un promedio movil

mydata2=df.copy()

''' Grafica'''

ax = mydata2[['MULTPL/SHILLER_PE_RATIO_MONTH - Value',
       'MULTPL/SP500_REAL_PRICE_MONTH - Value']].plot(secondary_y='MULTPL/SP500_REAL_PRICE_MONTH - Value')
ax2 = ax.twinx()
ax2.set_yscale('log')
plt.show()

ax = mydata2[['MULTPL/SHILLER_PE_RATIO_MONTH - Value',
       'MULTPL/SP500_DIV_YIELD_MONTH - Value']].plot(secondary_y='MULTPL/SP500_REAL_PRICE_MONTH - Value')
ax2 = ax.twinx()
ax2.set_yscale('log')
plt.show()

mydata2['MULTPL/SHILLER_PE_RATIO_MONTH - Value'].hist()
mydata2['MULTPL/SP500_DIV_YIELD_MONTH - Value'].hist()

#from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

#init_notebook_mode()

#iplot (
#mydata2['MULTPL/SHILLER_PE_RATIO_MONTH - Value'].values.tolist(),
#)
#mydata2['MULTPL/SHILLER_PE_RATIO_MONTH - Value'].iplot()

''' Pruebas'''

mydata3=mydata2.dropna()
X=mydata3['MULTPL/SHILLER_PE_RATIO_MONTH - Value']
y=mydata3['MULTPL/SP500_REAL_PRICE_MONTH - Value']
X=X.values.reshape(-1,1)
y=y.values.reshape(-1,1)

import numpy as np
from sklearn.linear_model import LinearRegression
#X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
# y = 1 * x_0 + 2 * x_1 + 3
#y = np.dot(X, np.array([1, 2])) + 3
reg = LinearRegression().fit(X, y)
reg.score(X, y)

reg.coef_

reg.intercept_ 

reg.predict(np.array([[16.64]]))

''' Lag y otros'''

#def transform(X,y):
#    #X=X.values.reshape(-1,1)
#    #y=y.values.reshape(-1,1)
#    reg = LinearRegression().fit(X, y)
#    return reg.score(X, y)


data=mydata2.copy()
data['spf']=data['MULTPL/SP500_REAL_PRICE_MONTH - Value'].shift(periods=-60) #Want lead
data['per_var']=data['spf']/data['MULTPL/SP500_REAL_PRICE_MONTH - Value']-1 
data=data.rename(columns={"MULTPL/SHILLER_PE_RATIO_MONTH - Value": "shiller_ratio"})
data['shiller_ma']=data['shiller_ratio'].rolling(12*50).mean()
data['div_ma']=data['MULTPL/SP500_DIV_YIELD_MONTH - Value'].rolling(12*50,min_periods=12*45).mean()
data['mean']=data['shiller_ratio']-data['shiller_ma']
data=data.rename(columns={"MULTPL/SP500_DIV_YIELD_MONTH - Value": "div_yield"})
data['mean_div']=data['div_yield']-data['div_ma']
#transform(data[['mean','MULTPL/SHILLER_PE_RATIO_MONTH - Value']],data['per_var'])


# Me parece que debe ser 0.60 porque es el acumulado de 5 años
data['id'] = data['per_var'].apply(lambda x: 1 if x > 0.60 else (0 if x<0 else 2))
#pre_data=data.copy()
pre_data=data.copy()
data = data[data['id']!= 2 ]
data=data.dropna()
vista=pre_data.head()

''' Train Test'''


from sklearn.model_selection import train_test_split
train,test=train_test_split(data,test_size=0.3,random_state=123)
print(len(train))
print(len(test))


''' Oversampling '''

from sklearn.utils import resample
import pandas as pd

train['id'].value_counts()

# Separate majority and minority classes
train_majority = train[train.id==1]
train_minority = train[train.id==0]

#len(df_majority)
 
# Upsample minority class
train_minority_upsampled = resample(train_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=len(train_majority),    # to match majority class
                                 random_state=123) # reproducible results
 
# Combine majority class with upsampled minority class
train_upsampled = pd.concat([train_majority, train_minority_upsampled])
 
# Display new class counts
train_upsampled.id.value_counts()
# 1    576
# 0    576
# Name: balance, dtype: int64



''' Modelo '''

import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.ensemble import RandomForestClassifier

train_upsampled=sm.add_constant(train_upsampled)
test=sm.add_constant(test)

#Logistic Regression

mod = smf.logit(formula='id ~  mean +  mean_div  ', data=train_upsampled)
res = mod.fit()
res.summary()

y_pred=res.predict(test)

# Random Forest

clf = RandomForestClassifier(n_estimators=100, max_depth=2,
                             random_state=0)


clf.fit(train_upsampled[['const','mean']], train_upsampled['id'])  

#y_pred=clf.predict_proba(test[['const','mean']])

#y_pred=y_pred[:,1]

''' ROC '''

from claudia import ks,gini

#print('----------------------------------------Train --------------------------------------')
#print(gini(y_traing,y_pred_traing,plot=True)) # Train
#print(ks(y_traing,y_pred_traing)) #Train

print('-----------------------------------------Test  -------------------------------------')
print(gini(test['id'],y_pred,plot=True)) # Test
print(ks(test['id'],y_pred)) #Test


''' Confusion Matrix '''

def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):
    plt.matshow(df_confusion, cmap=cmap) # imshow
    #plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    #plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)

from sklearn.metrics import confusion_matrix

test['pred']=y_pred
test['threshold']=y_pred
test['threshold']=test['threshold'].apply(lambda x: 1 if x > 0.50 else 0)

confusion=pd.DataFrame(confusion_matrix(test['id'], test['threshold']))

plot_confusion_matrix(confusion)

accuracy=(confusion.iloc[0,0]+confusion.iloc[1,1])/len(test)
print(accuracy)

''' Segmentando '''

pre_pred=res.predict(pre_data)

#pre_data=sm.add_constant(pre_data)
#pre_pred=clf.predict_proba(pre_data[['const','mean']])[:,1]

pre_data['pred']=pre_pred

col_names = {'count_nonzero': 'tasamalos', 'size': 'obs'}

pre_data['bucket'] = pd.qcut(pre_data['pred'], 12 ,\
         duplicates='drop',retbins=True)[0]

seg=pre_data.groupby('bucket')['id']\
.agg([np.mean, np.count_nonzero, np.size]).rename(columns=col_names)

seg2=pre_data.groupby('bucket')['per_var']\
.agg([np.mean,np.var])


def q1(x):
    return x.quantile(0.01)

def q2(x):
    return x.quantile(0.99)

f = {'number': [q1,q2]}

segg=pre_data.groupby('bucket')['per_var'].agg(f)

seg2.columns=['ren_pro','var_pro']
segg.columns=['VaR','-VaR']

seg3=seg.join(seg2[['ren_pro','var_pro']])
seg3=seg3.join(segg)

seg3['inverse_dispersion_index']=seg3['ren_pro']/seg3['var_pro']

seg3['R/Var']=seg3['ren_pro']/seg3['VaR']

print(seg3)

''' Prediction '''

mydata3=mydata2.copy()
mydata3['ma']=mydata3['MULTPL/SHILLER_PE_RATIO_MONTH - Value'].rolling(12*50).mean()
mydata3['mean']= mydata3['MULTPL/SHILLER_PE_RATIO_MONTH - Value']-mydata3['ma']
mydata3['ma_div']=mydata3['MULTPL/SP500_DIV_YIELD_MONTH - Value'].rolling(12*50,min_periods=12*45).mean()
mydata3['mean_div']= mydata3['MULTPL/SP500_DIV_YIELD_MONTH - Value']-mydata3['ma_div']
#mean=mydata3[['mean','mean_div']].iloc[-1,:]
#mean= np.array([1,mean])
#mean=mean.reshape(1,2)
#pred=pd.DataFrame(mean)
#pred.columns=['const','mean']
mydata3['pd']=res.predict(mydata3)

print(res.summary())

''' Correlacion '''
#Fuerte Corrleacion -- PCA?
train_upsampled[['mean','mean_div']].corr()


''' Graficos con plotly'''
#
#import plotly.plotly as py
#import plotly.graph_objs as go
#from plotly.offline import iplot
#
#import cufflinks
#
#mydata2['mean'].plot(kind='hist')
#
##pred=sm.add_constant(pred)
##pre_pred=clf.predict_proba(pred[['const','mean']])[:,1]





