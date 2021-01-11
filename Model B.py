# -*- coding: utf-8 -*-
"""
SFI CRT Foundations of Data Science
Machine Learning on Hydrodynamic Cavitation-Based Water Treatment Devices

Model B final code

@authors: Darragh Glavin, Eduardo Maekawa, Shane Mannion Stephen Mullins
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.regularizers import l1_l2
from ann_visualizer.visualize import ann_viz
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score


#load and read data
dir_name = os.path.dirname(os.path.realpath(__file__))     # Two versions included incase a line does not work on your device.
#dir_name = os.path.dirname(os.path.abspath("__file__"))     # .realpath() and/or .abspath() don't work on every device.

data = pd.read_csv(dir_name + '/Data.csv')

loss = 100

#repeat preprocess on original data for testing
data['n/dt']=data['n']/data['dt']
data['ln']=-np.log(-np.log(data['C/C0'])/(data['n']*(1+np.log(data['C/C0']))))*data['n']
data = data.fillna(0)

#split aug data into features and labels
dt =38 #train on 3 test on 1
features = data[(data.dt!=dt)].drop(['ln','C/C0'], axis=1)
labels = data[(data.dt!=dt)]['ln']
true = data[(data.dt==dt)]

#build neural network
while loss > 50:
    model = Sequential()
    model.add(Dense(2, input_dim=3, activation='relu',
                    bias_regularizer=l1_l2(0.001,0.01),
                    kernel_regularizer=l1_l2(0.001,0.01),kernel_initializer='normal',
                    bias_initializer='normal'))
    model.add(Dense(2,  activation='relu',bias_regularizer=l1_l2(0.001,0.01),
                    kernel_regularizer=l1_l2(0.001,0.01),kernel_initializer='normal',
                    bias_initializer='normal'))
    model.add(Dense(1,  activation='relu',bias_regularizer=l1_l2(0.001,0.01),
                    kernel_regularizer=l1_l2(0.001,0.01),kernel_initializer='normal',
                    bias_initializer='normal'))

    es = EarlyStopping(monitor='val_loss', mode='min', min_delta=1, verbose=1, patience=200)

    model.compile(loss='mean_absolute_error', optimizer='adam',metrics=["mae"])#, metrics=['r2'])
    #Fit the model and make predictions

    history = model.fit(features,labels,epochs=2000,validation_split=0.1,callbacks=[es])
    loss = history.history["val_mae"][-1]


pred = model.predict(true.drop(['ln','C/C0'],axis=1))
pred =pred.reshape(np.size(pred,0))
#convert linear form back original form
preds = np.exp(-true['n']*np.exp(-pred/true['n'])/(1+true['n']*np.exp(-pred/true['n'])))
preds = preds.fillna(1)

#plot loss
plt.title('Log Loss / Mean Squared Error',fontdict={'size':22})
plt.xlabel('Number of Epochs',fontdict={'size':18})
plt.ylabel('Log Loss',fontdict={'size':18})
plt.tick_params(axis='both', which='major', labelsize=14)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.yscale('log')
plt.legend()
plt.show()

#print error and R^2 sores for both log and original forms
model.summary()
print('\nlog R2 Score:',r2_score(y_true=true['ln'], y_pred=pred))
print('log Mean SQ Error:',mean_squared_error(y_true=true['ln'], y_pred=pred))
print('log Mean Abs Error:', mean_absolute_error(y_true=true['ln'], y_pred=pred))

print('\nR2 Score:',r2_score(y_true=true['C/C0'], y_pred=preds))
print('Mean SQ Error:',mean_squared_error(y_true=true['C/C0'], y_pred=preds))
print('Mean Abs Error:', mean_absolute_error(y_true=true['C/C0'], y_pred=preds))

ann_viz(model, title="Neural Network structure", view=False)

plt.title('Predictions versus Real log 38mm device Test data',fontdict={'size':16})
plt.xlabel('No. of passes',fontdict={'size':12})
plt.ylabel('C/C0',fontdict={'size':12})
plt.tick_params(axis='both', which='major', labelsize=8)
plt.plot(true['n'], true['ln'],'*',color='k', label='Real')
plt.plot(true['n'], pred,'.',color='r', label='Prediction')
plt.legend(prop={'size': 10})
plt.show()

plt.title('Predictions versus Real 38mm device Test data',fontdict={'size':16})
plt.xlabel('No. of passes',fontdict={'size':12})
plt.ylabel('C/C0',fontdict={'size':12})
plt.tick_params(axis='both', which='major', labelsize=8)
plt.plot(true['n'], true['C/C0'],'.',color='royalblue', label='Real')
sns.lineplot(true['n'], preds,color='orange', label='Prediction')
plt.legend(prop={'size': 10})
plt.show()

plt.title('Real and predicted device',fontdict={'size':16})
plt.xlabel('No. of passes',fontdict={'size':12})
plt.ylabel('C/C0',fontdict={'size':12})
plt.tick_params(axis='both', which='major', labelsize=8)
plt.plot(data['n'], data['C/C0'],'.',color='royalblue', label='Real')


datatest1 = pd.DataFrame({'dt':9,'n' : np.arange(0,700,50)})
datatest1['n/dt'] = datatest1['n']/datatest1['dt']
predictiontest1 = model.predict(datatest1)
predictiontest1 =predictiontest1.reshape(np.size(predictiontest1,0))
predictiontest1 = np.exp(-datatest1['n']*np.exp(-predictiontest1/datatest1['n'])/(1+datatest1['n']*np.exp(-predictiontest1/datatest1['n'])))
predictiontest1 = predictiontest1.fillna(1)
plt.plot(datatest1['n'], predictiontest1,color='g', label='9mm')

datatest2 = pd.DataFrame({'dt':15,'n' : np.arange(0,1000,50)})
datatest2['n/dt'] = datatest2['n']/datatest2['dt']
predictiontest2 = model.predict(datatest2)
predictiontest2 =predictiontest2.reshape(np.size(predictiontest2,0))
predictiontest2 = np.exp(-datatest2['n']*np.exp(-predictiontest2/datatest2['n'])/(1+datatest2['n']*np.exp(-predictiontest2/datatest2['n'])))
predictiontest2 = predictiontest2.fillna(1)
plt.plot(datatest2['n'], predictiontest2,color='r', label='15mm')

datatest3 = pd.DataFrame({'dt':25,'n' : np.arange(0,1100,50)})
datatest3['n/dt'] = datatest3['n']/datatest3['dt']
predictiontest3 = model.predict(datatest3)
predictiontest3 =predictiontest3.reshape(np.size(predictiontest3,0))
predictiontest3 = np.exp(-datatest3['n']*np.exp(-predictiontest3/datatest3['n'])/(1+datatest3['n']*np.exp(-predictiontest3/datatest3['n'])))
predictiontest3 = predictiontest3.fillna(1)
plt.plot(datatest3['n'], predictiontest3,color='y', label='25mm')

datatest4 = pd.DataFrame({'dt':50,'n' : np.arange(0,1400,50)})
datatest4['n/dt'] = datatest4['n']/datatest4['dt']
predictiontest4 = model.predict(datatest4)
predictiontest4 =predictiontest4.reshape(np.size(predictiontest4,0))
predictiontest4 = np.exp(-datatest4['n']*np.exp(-predictiontest4/datatest4['n'])/(1+datatest4['n']*np.exp(-predictiontest4/datatest4['n'])))
predictiontest4 = predictiontest4.fillna(1)
plt.plot(datatest4['n'], predictiontest4,color='k', label='50mm')

datatest5 = pd.DataFrame({'dt':100,'n' : np.arange(0,1500,50)})
datatest5['n/dt'] = datatest5['n']/datatest5['dt']
predictiontest5 = model.predict(datatest5)
predictiontest5 =predictiontest5.reshape(np.size(predictiontest5,0))
predictiontest5 = np.exp(-datatest5['n']*np.exp(-predictiontest5/datatest5['n'])/(1+datatest5['n']*np.exp(-predictiontest5/datatest5['n'])))
predictiontest5 = predictiontest5.fillna(1)
plt.plot(datatest5['n'], predictiontest5,color='m', label='100mm')

plt.legend()
plt.show()
