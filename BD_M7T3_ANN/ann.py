# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 18:37:30 2018

@author: alber
"""

# Librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math


###############################################################################

# Data Preparation

###############################################################################

# Dataset -> https://www.kaggle.com/ntnu-testimon/paysim1
#dataset = pd.read_csv('paysim.csv')
#dataset = dataset.sort_values(by='isFraud', ascending=False)
#dataset.reset_index(drop=True, inplace=True)
#d = dataset[dataset['isFraud']==1]
#dataset = dataset.iloc[0:len(d)*2]
#dataset.to_csv('paysim_reduced.csv')

dataset = pd.read_csv('paysim_reduced.csv')

# Data Preparation
df_aux = pd.get_dummies(dataset['type']).astype('int')
dataset.drop(['type', 'Unnamed: 0', 'nameOrig', 
              'nameDest', 'isFlaggedFraud', 'step',
              'newbalanceOrig', 'newbalanceDest'], axis=1, inplace=True)
dataset = dataset.join(df_aux)

X = dataset.loc[:, dataset.columns != 'isFraud'].values
y = dataset['isFraud'].values

# Train/Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


###############################################################################

# ANN Build

###############################################################################

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense


def create_nn(n_features, w_in, w_h1, n_var_out, optimizer, lr, momentum, decay):
    """
    Funcion para crear una NN para clasificacion binaria usando 2 HL
    
    """
    
    
    # Initialising the ANN
    model = Sequential()
    
    # First HL
    # [batch_size x n_features] x [n_features x w_in]
    model.add(Dense(units = w_in, input_dim = n_features, 
                    kernel_initializer = 'normal', 
                    activation = 'relu')) 
    # Second HL
    # [batch_size x w_in] x [w_in x w_h1]
    model.add(Dense(units = w_h1, input_dim = w_in, 
                    kernel_initializer = 'normal', 
                    activation = 'relu'))
    
    # Output Layer
    # [batch_size x w_h1] x [w_h1 x w_out]
    model.add(Dense(units = n_var_out, 
                    kernel_initializer = 'normal', 
                    activation = 'sigmoid')) 
    
    # Compile Model
    # Loss Function -> Cross Entropy (Binary)
    # Optimizer -> sgd, adam...
    if optimizer == 'sgd':
        keras.optimizers.SGD(lr=lr, momentum=momentum, decay=decay, nesterov=False)
        model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    else:
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
    return model



## fix random seed for reproducibility
#from tensorflow import set_random_seed
#from numpy.random import seed
#value = 7
#seed(value)
#set_random_seed(value)


# Parametros
n_features = np.shape(X_train)[1]
w_in = 12
w_h1 = 8
n_var_out = 1
batch_size = 100
nb_epochs = 100
optimizer = 'adam'
lr = 0.1
momentum = 0.01
decay = 0.0

# Create NN
model = create_nn(n_features, w_in, w_h1, n_var_out, optimizer, lr, momentum, decay)
    
# Fitting the ANN to the Training set
model.fit(X_train, y_train, batch_size = batch_size, epochs = nb_epochs)

    
###############################################################################

# ANN Predictions

###############################################################################

# Predict
y_pred = model.predict(X_test)
y_pred = np.round(y_pred)

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)












