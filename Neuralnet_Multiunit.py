# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# checking for different units [20 100 200 500]
import numpy as np
import random 
import csv 
import matplotlib.pyplot as plt

def init_network(n_layers, n_units):
    
    h1 = n_units[n_layers]
    h2 = n_units[n_layers+1]
    
    num = np.sqrt(6)
    den = np.sqrt(h1+h2)
    b = num/den
    
    w = np.random.uniform(low =-b, high= b,size=(h2,h1))
    BIAS = np.zeros([h2,1])
    
    return w, BIAS

def linear(X,W,b):
    if np.isnan(W.any()) == True:
        print ('W driven to infinity' )
    afirst = np.dot(W,X)

    a = afirst + b
    
    return a

def sigmfwd(a):
    ex = np.exp(-a)
    num = np.ones([hidden,1])
    denometer = 1 + ex
    output = np.divide(num,denometer*1.0)
    return output

def softmax(h):
#    e_x = np.exp(h - np.max(h))
#    return e_x / e_x.sum()
    e = np.exp(h)
    den = np.sum(e,axis = 0,keepdims = True)
    
    output = e/den
    return output
    

def predict(prob):
    return np.argmax(prob)
		
def CrossEntropy(SM,yh):
    p = np.array([1,1])
    p = SM[yh]
    C = -np.log(p)
    return C 

def Update(w1,w2,eta,d1,d2,b1,b2,db1,db2,m,v1,v2,vb1,vb2):
    v1 = (m * v1) - (eta * d1)
    v2 = (m * v2) - (eta * d2) 
    vb1 = (m*vb1) - (eta * db1)
    vb2 = (m * vb2) - (eta * db2)
    w1 = w1+ v1 
    w2 = w2+ v2 
    b1 = b1 + vb1 
    b2 = b2 + vb2
    return w1,w2 ,b1,b2,v1,v2,vb1,vb2  

def backward_softmax(p,y):
    Doutput = -(y-p)    
    return Doutput

def backward_sigmoid(Dprev,iput):
    output = iput * (1-iput)
    D = np.multiply(Dprev, output)
    return D

def backward_innerproduct(Dprevv,iput,w):
    Dinput = np.dot(np.transpose(w) , Dprevv)
    Dbias = 1*Dprevv
    Dweights =np.dot( Dprevv,np.transpose(iput))
    return Dinput,Dbias,Dweights

def Predict_validation(ip_imgV,y, W1,W2,bias1,bias2):
    c = 0;
    ls = 0;
    for i in range (0,ip_imgV.shape[0]-1):
        X_Valid = np.reshape(ip_imgV[i,:],(784,1))
        A1_v = linear(X_Valid,W1,bias1)
        S1_v = sigmfwd(A1_v)
        A2_v = linear(S1_v,W2,bias2)
        Softmax_v = softmax(A2_v)
        valid_pred = np.argmax(Softmax_v)
        ls = ls - np.log(Softmax_v[int(y[i])])
        if valid_pred != y[i]:
            c = c+1
    value = c/ip_imgV.shape[0]
    l = ls/ip_imgV.shape[0]
    return (value)   ,l 
    

layers = 2;
units  = [784, 100 ,10]
hidden = units[1]
ipValid = np.loadtxt(open("/Users/dhanashreebalaram/Downloads/digitsvalid.txt", "rb"), delimiter=",", skiprows=0)
ip2 = np.loadtxt(open("/Users/dhanashreebalaram/Downloads/digitstrain.txt", "rb"), delimiter=",", skiprows=0)
iptest = np.loadtxt(open("/Users/dhanashreebalaram/Downloads/digitstest.txt", "rb"), delimiter=",", skiprows=0)
ip = np.random.permutation(ip2)
ipV = np.random.permutation(ipValid)
ipT = np.random.permutation(iptest)
# extract the values of labels from input
labels = ip[:,ip.shape[1]-1]
labelsV = ipV[:,ipV.shape[1]-1]
labelsT = ipT[:,ipT.shape[1]-1]

ntrain =ip.shape[0]
ntest =ipT.shape[0]
nvalid =ipV.shape[0]
Ytrue = np.reshape(labels,(3000,1))
y_labelV = np.reshape(labelsV,(nvalid,1))
y_labelT = np.reshape(labelsT,(ntest,1))

ip_imgV = ipV[:,0:ipV.shape[1]-1]
ip_img = ip[:,0:ip.shape[1]-1]
ip_imgT = ipT[:,0:ipT.shape[1]-1]
epoc = 100

W1 , bias1 = init_network(0,units)

W2, bias2 =  init_network(1, units)
Loss = 0
TrainLoss_cat = []
ValLoss_cat = []
TrainError_cat = []
ValError_cat =[]
TestLoss_cat =[]
TestError_cat =[]
dw1store = [0]
dw2store = [0]
mu = 0.5 
lam = 0.0001  
LearningRate = 0.01     
v1= np.zeros([hidden ,784])
v2= np.zeros([10 ,hidden])
vb1= np.zeros([hidden, 1])
vb2= np.zeros([10, 1])

for epochs in range (epoc):
    cn=0
    Loss =0
    L=0
    
    for i in range (0,2999):
        inp = ip_img[i,:]
        X_train = np.reshape(inp,(784,1))
        
        A1 = linear(X_train,W1,bias1)
#        print ('\nA1 - input that goes into pre activation 1\n',A1)
        
        if np.isnan(A1.any()):
            print ('A1 driven to infinity... ')
        S1 = sigmfwd(A1)
        if np.isnan(S1.any()):
            print ('S1 driven to infinity... ')
        A2 = linear(S1,W2,bias2)
        if np.isnan(A2.any()):
            print ('A2 driven to infinity... ')
        Softmax = softmax(A2)
        if (Softmax.any() < 0):
            print ('Softmax driven to negative values... ')
#        temp = E/DEN
#        print ('-------------------------------------------------------------')
#        print ('Iteration number ', i)
#        print ('\nSoftmax \n ', Softmax)
        
        Ypredict = np.argmax(Softmax) #class number
        Yhot = np.zeros([10,1])
        Yhot[int(Ytrue[i])] = 1
        
        CrossEntr = CrossEntropy(Softmax, int(Ytrue[i]))
#        print (CrossEntr)
        Loss = Loss + CrossEntr
       
        
        
        delta1 = backward_softmax(Softmax,Yhot)
#        print ('\ndelta1 \n', delta1)
        if np.isnan(delta1.any()):
            print ('Delta of softmax going to infinity....')
        delta2,delta2b,delta2W = backward_innerproduct(delta1,S1,W2)
        delta3 = backward_sigmoid(delta2,S1)
        delta4,delta4b,delta4W = backward_innerproduct(delta3,X_train,W1)
#        print ('\nDelta of L wrt Weights2 \n',delta2W)
#        print ('\nDelta for sigmoid \n' , delta3)
#        print ('\nDelta wrt inputs \n' , delta4)
#        print ('\nDelta wrt w1 \n', delta4W)
#        print ('\nDelta wrt b1 \n', delta4b)
#        print ('\nweights 1 \n', W1)
#        print ('\nWeights 2 \n', W2)
       
        if Ypredict != Ytrue[i]:
            cn = cn+1
        
        W1,W2,bias1,bias2,V1,V2,Vb1,Vb2 = Update(W1,W2,LearningRate,delta4W,delta2W,bias1,bias2,delta4b,delta2b,mu,v1,v2,vb1,vb2)
        v1 = V1
        v2 = V2
        vb1 = Vb1
        vb2 = Vb2
    
        
   
    
    Verror,Vloss = Predict_validation(ip_imgV,y_labelV, W1,W2,bias1,bias2)
    TestError,TestLoss = Predict_validation(ip_imgT,y_labelT, W1,W2,bias1,bias2)
    Ls = (Loss/3000)
    TrainError = (cn/3000) 
    print ('------------------------------------------------------------------')
    print ('Epoch :' ,epochs)
    print ('Train classification Acc ',1-TrainError)
#    print ('Train Loss',Ls)
    print ('Validation Classification Acc ', 1-Verror)
#    print ('Validation Loss ',Vloss)
    print ('Test Classification Acc',1-TestError)
#    print ('Test Loss ',TestLoss)
    
    TrainLoss_cat.append(Ls)
    ValLoss_cat.append(Vloss)    
    TrainError_cat.append(TrainError)
    ValError_cat.append(Verror)
    TestLoss_cat.append(TestLoss)    
    TestError_cat.append(TestError)
    
#Plotting test,train and validation loss graphs    
p1,=plt.plot(range(0,epochs+1),TrainLoss_cat)
plt.ylabel('Training Cross Entropy Loss')
plt.xlabel('Epochs')
plt.title('Cross Entropy Vs Epochs')

p2,=plt.plot(range(0,epochs+1),ValLoss_cat)
plt.ylabel('Validation Cross Entropy Loss')
plt.xlabel('Epochs')

p3,=plt.plot(range(0,epochs+1),TestLoss_cat)
plt.ylabel('Testing Cross Entropy Loss')
plt.xlabel('Epochs')
#plt.legend([p1,p2,p3],['Train Error','Validation Error','Train Error']) 
plt.legend([p1,p2,p3],['Train Error','Validation Error','Test Error'])  
plt.show()

#
#plt.plot(range(0,epochs+1),TrainError_cat)
#plt.ylabel('Training classification error')
#plt.xlabel('Epochs')
#plt.show()
#plt.plot(range(0,epochs+1),ValError_cat)
#plt.ylabel('Validation classification error')
#plt.xlabel('Epochs')
#plt.show()
#plt.plot(range(0,epochs+1),TestError_cat)
#plt.ylabel('Testing classification error')
#plt.xlabel('Epochs')
#plt.show()    
