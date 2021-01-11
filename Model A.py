# -*- coding: utf-8 -*-
"""
SFI CRT Foundations of Data Science
Machine Learning on Hydrodynamic Cavitation-Based Water Treatment Devices

Model A final code

@authors: Darragh Glavin, Eduardo Maekawa, Shane Mannion and Stephen Mullins
"""

#===== Import Libraries =====#
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#===== Data Reading =====#
# Imports the dataset which should be in the same directory as the code

#dir_name = os.path.dirname(os.path.realpath(__file__))     # Two versions included incase a line does not work on your device.
dir_name = os.path.dirname(os.path.abspath("__file__"))     # .realpath() and/or .abspath() don't work on every device.

df = pd.read_csv(dir_name + '/Data.csv')

#===== Data pre-processing =====#


# Synthetic Zeros used to try encourage the model to predict C/C0=1 when n=0
dt38_0 = np.array([38, 0, 1])
zeros = np.row_stack((dt38_0,dt38_0,dt38_0,dt38_0,dt38_0,dt38_0,dt38_0,dt38_0,dt38_0,dt38_0))

zerosDF = pd.DataFrame(zeros, columns=['dt', 'n', 'C/C0'])   
df = df.append(zerosDF)


# Transform the output
df['C/C0']=np.log(df['C/C0'])+1 

# Normalise the dataset
df['n'] = df['n']/1114.1
df['dt'] = df['dt']/38

# Create a new feature to improve predictions
df['n/dt'] = df['n']/df['dt']

# Seperate the dataset into inputs (features), outputs (labels). 
labels = df["C/C0"]
features = df.drop(["C/C0"], axis=1)

# Divide dataset into mixed Train, Validation and Test sets
x_train, x_test, y_train, y_test = train_test_split(features,labels, test_size = 0.3, random_state = 42, shuffle=True, stratify=df["dt"])
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size = 0.5, random_state = 42, shuffle= True)


#===== Machine Learning Architecture =====#
from keras.models import Sequential
from keras.layers import Dense

# Build the model
model = Sequential()
model.add(Dense(2, input_dim=3, activation = 'sigmoid')) # Input layer & First hidden layer
model.add(Dense(1)) # Second Hidden Layer
model.add(Dense(1, activation='sigmoid')) # Output layer
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae','mse'] )

# Fit the model
history = model.fit(x_train, y_train, epochs=10000, validation_data=(x_val, y_val))

# Check the model structure and the number of free parameters
model.summary()

#===== Evaluate the model =====#
from sklearn import metrics

# Training set scores
y_train = np.exp(y_train - 1) # Transform the labels back to exp curve form
train_pred = np.exp(model.predict(x_train)-1) # Transform the predictions to exp curve form

print("\n\nTRAINING SCORES:")
print('Training - Mean Absolute Error regression loss (MAE): %.6f' %metrics.mean_absolute_error(y_train, train_pred))
print('Training - Mean squared error regression loss (MSE): %.6f' % metrics.mean_squared_error(y_train, train_pred))
print('Training - R^2 (coefficient of determination) regression score function: %.6f' %metrics.r2_score(y_train, train_pred))

# Validation set scores
y_val = np.exp(y_val - 1) # Transform the labels back to exp curve form
val_pred = np.exp(model.predict(x_val)-1) # Transform the predictions to exp curve form
print("\n\nVALIDATION SCORES:")
print('Validation - Mean Absolute Error regression loss (MAE): %.6f' %metrics.mean_absolute_error(y_val,val_pred))
print('Validation - Mean squared error regression loss (MSE): %.6f' % metrics.mean_squared_error(y_val, val_pred))
print('Validation - R^2 (coefficient of determination) regression score function: %.6f' %metrics.r2_score(y_val, val_pred))


# Plot the Training and Validation MSE Loss curves
plt.title('Loss / Mean Squared Error',fontdict={'size':22})
plt.xlabel('Number of Epochs',fontdict={'size':18})
plt.ylabel('Loss',fontdict={'size':18})
plt.tick_params(axis='both', which='major', labelsize=14)
plt.yscale('log')
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.legend()
plt.show()

#Plot real vs prediction for Training set
plt.title('Predictions vs Real values on Train data',fontdict={'size':22})
plt.xlabel('No. of passes',fontdict={'size':18})
plt.ylabel('C/C0',fontdict={'size':18})
plt.tick_params(axis='both', which='major', labelsize=14)
plt.plot(x_train['n'], y_train,'.',color='royalblue', label='Real')
plt.plot(x_train['n'], train_pred,'.',color='darkorange', label='Predictions')
plt.legend(prop={'size': 10})
plt.show()


#Plot real vs prediction for Validation set
plt.title('Predictions vs Real values on Validation data',fontdict={'size':22})
plt.xlabel('No. of passes',fontdict={'size':18})
plt.ylabel('C/C0',fontdict={'size':18})
plt.tick_params(axis='both', which='major', labelsize=14)
plt.plot(x_val['n'], y_val,'.',color='royalblue', label='Real')
plt.plot(x_val['n'], val_pred,'.',color='darkorange', label='Predictions')
plt.legend(prop={'size': 10})
plt.show()



#===== Test the model =====#

# Test set scores
y_test = np.exp(y_test - 1) # Transform the labels back to exp curve form
test_pred = np.exp(model.predict(x_test)-1) # Transform the predictions to exp curve form
print("\n\nTEST SCORES:")
print('Test - Mean Absolute Error regression loss (MAE): %.6f' %metrics.mean_absolute_error(y_test,test_pred))
print('Test - Mean squared error regression loss (MSE): %.6f' % metrics.mean_squared_error(y_test, test_pred))
print('Test - R^2 (coefficient of determination) regression score function: %.6f\n' %metrics.r2_score(y_test, test_pred))


#Plot real vs prediction for Test set
plt.title('Predictions versus Real values on Test data',fontdict={'size':18})
plt.xlabel('No. of passes',fontdict={'size':18})
plt.ylabel('C/C0',fontdict={'size':18})
plt.tick_params(axis='both', which='major', labelsize=14)
plt.plot(x_test['n'], y_test,'.',color='royalblue', label='Real')
plt.plot(x_test['n'], test_pred,'.',color='darkorange', label='Predictions')
plt.legend(prop={'size': 10})
plt.show()

# Generate Synthetic Test feature data (dt, n & n/dt) for model Interpolation and Extrapolation predictions
import math
synthDF = pd.DataFrame(columns = ['dt', 'n', 'C/C0'])

# Function for Equation 6 from the paper
def genNewDiameterData(phi_inf, beta, dt, n):
    phi0 = phi_inf * math.exp(beta/dt)
    c_c0 = np.array([]) 
    
    for i in n:
        ans = math.exp( (-phi0*i) / (1 + phi0*i) )
        c_c0 = np.append(c_c0, ans) 
        
    dt = [dt]*len(n)    
    synData = np.column_stack((dt,n,c_c0))
    return synData

# Pparameters obtained from the paper
phi_inf = 0.00015
beta = 10.85

for i in [9,15,50]:
    if i==9:
        max = 760
        step = 60
    elif i==15:
        max=1100
        step=100
    else:
        max = 1400
        step = 100

    n = np.arange(0,max,step)
    
    synth  =np.round(genNewDiameterData(phi_inf, beta, i, n),6)
    synth = pd.DataFrame(synth, columns=['dt', 'n', 'C/C0'])
    synthDF = synthDF.append(synth)

# Normalise the synthetic data as was done for the real data above
synthDF['n'] = synthDF['n']/1114.1
synthDF['dt'] = synthDF['dt']/38
synthDF['n/dt'] = synthDF['n']/synthDF['dt']

# Seperate the synthetic data individually and into their respective features and labels
feats_9 = synthDF[(synthDF['dt']==9/38)].drop(['C/C0'],axis=1)
feats_15 = synthDF[(synthDF['dt']==15/38)].drop(['C/C0'],axis=1)
feats_50 = synthDF[(synthDF['dt']==50/38)].drop(['C/C0'],axis=1)

label_9 = synthDF[(synthDF['dt']==9/38)]['C/C0']
label_15 = synthDF[(synthDF['dt']==15/38)]['C/C0']
label_50 = synthDF[(synthDF['dt']==50/38)]['C/C0']

# Model predictions of 9mm, 15mm and 50mm
interpolation_test_9 = np.exp(model.predict(feats_9) - 1)   # Remember the model learned with transformed label
interpolation_test_15 = np.exp(model.predict(feats_15) - 1) # data so will predict transformed data, we need to
extrapolation_test_50 = np.exp(model.predict(feats_50) - 1) # transform the predictions back into the exp form.

# Transform the Observed Data back into exp form for plot
df['C/C0'] = np.exp(df['C/C0'] - 1)

# Plot the Interpolation and Extrapolation predictions
plt.title('Interpolation/Extrapolation and Observed Data',fontdict={'size':18})
plt.xlabel('No. of passes',fontdict={'size':18})
plt.ylabel('C/C0',fontdict={'size':18})
plt.tick_params(axis='both', which='major', labelsize=14)
plt.plot(feats_9['n'], interpolation_test_9,'-',color='royalblue', label='Interpolation - 9mm')
plt.plot(feats_15['n'], interpolation_test_15,'-',color='purple', label='Interpolation - 15mm')
plt.plot(feats_50['n'], extrapolation_test_50,'-',color='red', label='Extrapolation - 50mm')
plt.plot(df['n'], df['C/C0'], '.', color = 'darkorange', label = 'Real Observations')
plt.legend(prop={'size': 10})
plt.show()

#Interpolation and Extrapolation predictions compared with the equation 6
plt.title("NN prediction vs Paper's Equation",fontdict={'size':18})
plt.xlabel('No. of passes',fontdict={'size':18})
plt.ylabel('C/C0',fontdict={'size':18})
plt.tick_params(axis='both', which='major', labelsize=14)
# Plot paper's equation curves
plt.plot(feats_9['n'], label_9,'--', color='grey', label='Eq.')
plt.plot(feats_15['n'], label_15,'--', color='grey')
plt.plot(feats_50['n'], label_50,'--', color='grey')

plt.plot(feats_9['n'], interpolation_test_9,'-',color='royalblue', label='9mm')
plt.plot(feats_15['n'], interpolation_test_15,'-',color='purple', label='15mm')
plt.plot(feats_50['n'], extrapolation_test_50,'-',color='red', label='50mm')
plt.plot(df['n'], df['C/C0'], '.', color = 'darkorange', label = 'Real')

plt.legend(prop={'size': 10})
plt.show()