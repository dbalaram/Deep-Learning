#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 19:25:34 2017

@author: dhanashreebalaram
"""

import numpy as np
import random 
import matplotlib.pyplot as plt
from matplotlib import gridspec

#Storing the dataset
data_Train = np.loadtxt(open("/Users/dhanashreebalaram/Downloads/digitstrain.txt", "rb"), delimiter=",", skiprows=0)
data_Valid = np.loadtxt(open("/Users/dhanashreebalaram/Downloads/digitsvalid.txt", "rb"), delimiter=",", skiprows=0)
data_Test  = np.loadtxt(open("/Users/dhanashreebalaram/Downloads/digitstest.txt", "rb"), delimiter=",", skiprows=0)

#Assigning the 784 elements into Train, Test and Valid

Valid = data_Valid[:,0:data_Valid.shape[1]-1]
Test = data_Test[:,0:data_Test.shape[1]-1]

#Assigning the labels 
labels_Train = data_Train[:,data_Train.shape[1]-1]
labels_Valid = data_Valid[:,data_Valid.shape[1]-1]
labels_Test = data_Test[:,data_Test.shape[1]-1]

def Sigmoid(inp):
     
     value = np.exp(-inp)
     return 1/(1+value)
 
def Sigmoid_backward(out,delta):
    Value = out * (1-out)
    return Value * delta

def Loss_backward(x,x_bar):
    return(x_bar-x)

def backward_innerproduct(Dprevv,iput,w):
    Dinput = np.dot( Dprevv,w.T)
    Dbias = 1*Dprevv
    Dweights =np.dot(iput.T,Dprevv)
    return Dinput,Dbias,Dweights

def CrossEntropy(x,x_bar):
    Loss = -(x * np.log(x_bar))-((1-x) * np.log(1-x_bar))
    CE = np.sum(Loss)
    return CE

def initialize():
    hidden_no = 100;
    visible_no = Train.shape[1];
    W1 = np.random.normal(0, .1, [visible_no,hidden_no])
    W2 = np.random.normal(0, .1, [hidden_no,visible_no])
    C = np.zeros([1,visible_no])
    B = np.zeros([1,hidden_no])
    return hidden_no,visible_no,W1,W2,B,C

def Update(W1,W2,B,C,deltaW1,deltaW2,deltaC,deltaB,alpha):
    W1 -= alpha * deltaW1
    W2 -= alpha * deltaW2
    B -= alpha*deltaB
    C -= alpha*deltaC
    return W1,W2,B,C

def PredictError(dataset,W1,W2,B,C):
    V_error = 0
    for k in range(dataset.shape[0]):
        v = dataset[k,0:784]
        V = np.reshape(v,(1,784))
        a1 = np.dot(V,W1) + B
        h = Sigmoid(a1)
        a2 = np.dot(h,W2) + C
        v_hat = Sigmoid(a2)
        V_error += CrossEntropy(V,v_hat) 
        
    Error_Val = V_error/1000
    return Error_Val    

    
epoch = 100;
hidden_no,visible_no,W1,W2,B,C = initialize()

alpha = 0.01
Train_error =[]
Validation_error =[]
Train_error2 =[]
Error=0


for i in range(epoch):
    Train = np.random.permutation(data_Train[:,0:data_Train.shape[1]-1])
    Error = 0
    V_error = 0
    for j in range(Train.shape[0]):
        x = Train[j,0:784]
        X = np.reshape(x,(1,784))
        #Start: forward
        #preactivation
        A1 = np.dot(X,W1) + B;
        #hidden-forward 
        H = Sigmoid(A1)
        #reconstruction preactivation
        A2 =  np.dot(H,W2) + C;
        #reconstruction non lineaar
        X_hat = Sigmoid(A2)
#        end:Forward
        
       
        
        #start:backward
        deltaL = Loss_backward(X,X_hat);
        deltaH,deltaC,deltaW2 = backward_innerproduct(deltaL,H,W2)
        delta1 = Sigmoid_backward(H,deltaH)
        deltaX,deltaB,deltaW1 = backward_innerproduct(delta1,X,W1)
        #end: backward
        W1,W2,B,C = Update(W1,W2,B,C,deltaW1,deltaW2,deltaC,deltaB,alpha)
        Error += CrossEntropy(X,X_hat)  
        
    E = Error/3000    
    Train_error.append(E)

    E_V = PredictError(Valid,W1,W2,B,C)
    Validation_error.append(E_V)
    print('Epoch: ',i,' Valid_Loss: ', E_V, 'Train_Loss ', E)
    
p1,=plt.plot(range(epoch),Train_error)        
p2,=plt.plot(range(epoch),Validation_error)         
plt.legend([p1,p2],['Train Error','Validation Error'])          

indx = 0
row = 10
col = 10
fig = plt.figure(figsize=(col+1, row+1)) 

GridSp = gridspec.GridSpec(row, col,
         wspace=0.0, hspace=0.0, 
         top=1.-0.5/(row+1), bottom=0.5/(row+1), 
         left=0.5/(col+1), right=1-0.5/(col+1)) 

for i in range(row):
    for j in range(col):
        im = np.reshape(W1[:,indx],(28,28))
        s= plt.subplot(GridSp[i,j])
        s.imshow(im)
        s.set_xticklabels([])
        s.set_yticklabels([])
        indx += 1
plt.show()  


indx = 0
row = 10
col = 10
fig = plt.figure(figsize=(col+1, row+1)) 

GridSp = gridspec.GridSpec(row, col,
         wspace=0.0, hspace=0.0, 
         top=1.-0.5/(row+1), bottom=0.5/(row+1), 
         left=0.5/(col+1), right=1-0.5/(col+1)) 

for i in range(row):
    for j in range(col):
        im = np.reshape(W2[indx,:],(28,28))
        s= plt.subplot(GridSp[i,j])
        s.imshow(im)
        s.set_xticklabels([])
        s.set_yticklabels([])
        indx += 1
plt.show()   