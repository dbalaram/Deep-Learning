#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 23:15:00 2017

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
Train = data_Train[:,0:data_Train.shape[1]-1]
#Assigning the labels 
labels_Train = data_Train[:,data_Train.shape[1]-1]
labels_Valid = data_Valid[:,data_Valid.shape[1]-1]
labels_Test = data_Test[:,data_Test.shape[1]-1]

def sigma(inp,w,bias):
    Val=np.dot(inp,w)
    value = np.exp(-Val-bias)
    return 1/(1+value)

def initialize():
    hidden_no = 100;
    visible_no = Train.shape[1];
    W = np.random.normal(0, 0.1, [visible_no,hidden_no])
    A = np.zeros([1,visible_no])
    B = np.zeros([1,hidden_no])
    return hidden_no,visible_no,W,A,B

def update(pos,neg,h_old,h_new,x_old,x_new,w,b,a,alpha):
    w += alpha*(pos-neg)
    b += alpha*(h_old - h_new)
    a += alpha*(x_old- x_new)
    return w,a,b

def crossEntropy(inp1,inp2):
    Loss = -(inp1 * np.log(inp2))-((1-inp1) * np.log(1-inp2))
    CE = np.sum(Loss)
    return CE

def CalcError(dataset,w,a,b):
    e=0
    for j in range (dataset.shape[0]):
        x = dataset[j,:]
        x = np.reshape(x,[1,784])
        xo = x;
        
        for k in range(K):
            P1 = sigma(x,w,b);
            h = np.random.binomial(1,P1,[P1.shape[0],P1.shape[1]])
            Px = sigma(h,w.T,a);
            x = np.random.binomial(1,Px,[x.shape[0],x.shape[1]])
            
        Xnew = Px    
        e += crossEntropy(xo,Xnew)
    Err = e/ dataset.shape[0]   
    return Err
            
epoch = 70;
K = 1

hidden_no,visible_no,W,A,B = initialize()
    
alpha = 0.01
Train_error =[]
Validation_error =[]
   
Error =0
    
for i in range(epoch):
    #Train = np.random.permutation(data_Train[:,0:data_Train.shape[1]-1])
    
    Error = 0;
    for j in range(Train.shape[0]):
            
            x = Train[j,0:784]
            
            X = np.reshape(x,(1,784))
            Xold = X;
            P = sigma(X,W,B);
            for k in range(K):
                
                PhX = sigma(X,W,B)
                H = np.random.binomial(1,PhX,[X.shape[0],hidden_no])
                
                PXh = sigma(H,W.T,A)
                
                X = np.random.binomial(1,PXh,[X.shape[0],X.shape[1]])
                
                
                
                
            X_bar =  PXh
            PhX_bar = sigma(X_bar,W,B)
            pos = np.dot(Xold.T,P)
            neg  = np.dot(X_bar.T,PhX_bar)
                
                
                #calling the update function
            W,A,B = update(pos,neg,P,PhX_bar,Xold,X_bar,W,B,A,alpha)
                
            Error += crossEntropy(Xold,X_bar)
                #        Error += np.sum(-(X * np.log(X_bar))-((1-X) * np.log(1-X_bar)
                
                
    E = Error/3000
                
    Train_error.append(E)  
    E_V = CalcError(Valid,W,A,B)
    Validation_error.append(E_V)
    print('Epoch: ',i , 'Train error:',E, '   Valid error:',E_V) 

p1,=plt.plot(range(epoch),Train_error)        
p2,=plt.plot(range(epoch),Validation_error)  
plt.xlabel('Epochs')
plt.ylabel('Cross Entropy')
plt.title('Cross Entropy Vs Epochs')       
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
        im = np.reshape(W[:,indx],(28,28))
        s= plt.subplot(GridSp[i,j])
        s.imshow(im,cmap = 'gray')
        s.set_xticklabels([])
        s.set_yticklabels([])
        indx += 1
plt.show()


# 5c) Sampling from RBM Model    
x_bar = np.zeros([100,784])    
for i in range(100):
    x = np.random.uniform(0,1,[1,784])
    for k in range(1000):
        P1 = sigma(x,W,B);
        h = np.random.binomial(1,P1,[1,100])
        P2 = sigma(h,W.T,A)
        x = np.random.binomial(1,P2,[1,784])
        
    x_bar[i,:] = P2  
    
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
        im = np.reshape(x_bar[indx,:],(28,28))
        s= plt.subplot(GridSp[i,j])
        s.imshow(im,cmap = 'gray')
        s.set_xticklabels([])
        s.set_yticklabels([])
        indx += 1
plt.show()            
    
      