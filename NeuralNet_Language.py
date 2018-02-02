#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 00:43:24 2017

@author: dhanashreebalaram
"""

import numpy as np
from collections import Counter
from collections import OrderedDict
import operator
import matplotlib.pyplot as plt
import pickle

Analysis = True
Visualize = True
tanh = False
hidden = 128         
def preprocessing():          
    fp = open('/Users/dhanashreebalaram/Desktop/train.txt', 'r')
    data = fp.read()
    D = data.lower().split()
    fp.close()  
    c=0
    Dict ={}  
    for word in D:
        if word not in Dict:
    #        print(1)
            Dict[word] = 0
        Dict[word] += 1     
       
    Dict = sorted(Dict.items(), key=operator.itemgetter(1), reverse = True)

    W = []
    with open('/Users/dhanashreebalaram/Desktop/train.txt') as file:
        for line in file:
            words = line.lower().split()
            words.insert(0,'START')
            words.insert(len(words),'END')
            c = c+1
            W.append(words)
    
        
    unk=0
    #now Dict contains keys --> words and the frequency of them occurring next to it.       
    dictionary = Dict[0:7997]  
    unk= len(Dict)- 7997 
    dictionary.append(('START',c))   
    dictionary.append(('END',c))  

    
    For_grams = []       
    for sublist in W:
        for item in sublist:
            if item not in dict(dictionary):
                item = 'UNK'
                unk += 1
            For_grams.append(item)    
    
     
    G=[]   
    for i in range (0,len(For_grams)-4):
        l = For_grams[i:i+4]
        G.append(l)
           
    Gram= []
    for j in range(len(G)):
        if G[j][0]!= 'END' and G[j][1]!= 'END' and G[j][2]!= 'END':
            Gram.append(G[j])
    
    dictionary.append(('UNK', unk) )
    
    Gram_count=Counter(tuple(sublist) for sublist in Gram )
    
    #For validation
    Val = []
    with open('val.txt') as file:
        for line in file:
            words = line.lower().split()
            words.insert(0,'START')
            words.insert(len(words),'END')
            Val.append(words)
    
    Val_g=[]
    for sublist in Val:
        for item in sublist:
            if item not in dict(dictionary):
                item = 'UNK'
            Val_g.append(item) 
            
    Val_Gram = []
    for i in range (0,len(Val_g)-4):
        l = Val_g[i:i+4]
        Val_Gram.append(l)   
    
    Gram_Val = []
    for j in range(len(Val_Gram)):
        if Val_Gram[j][0]!= 'END' and Val_Gram[j][1]!= 'END' and Val_Gram[j][2]!= 'END':
            Gram_Val.append(Val_Gram[j])
            
    # Plotting graoh of counts of four_grams        
    x = list(Gram_count.values())
    plt.plot(x)
    plt.ylabel('Frequency of occurrence (Counts)')
    plt.xlabel('Four Grams')
    plt.title('Plot of counts of Four-Grams ')
    

    return Gram, dictionary, Gram_Val

def initialize():
    WE= np.random.normal(0,0.1,[8000 ,16]) # word embeddings
    W1 = np.random.normal(0,0.1,[48 ,hidden]) # weight vector 1
    W2 =  np.random.normal(0,0.1,[hidden ,8000]) # weight vector 1
    b1=  np.random.normal(0,0.1,[1 ,hidden])
    b2 = np.random.normal(0,0.1,[1 ,8000])
    return WE, W1,W2,b1,b2
    
def CalcL(we,F,W1,W2,bias1,bias2):
#    print('Calculating   Loss')
    index = []
    out = []
    
    for i in range(len(F)):
        index.append([vocab.index(F[i][0]),vocab.index(F[i][1]),vocab.index(F[i][2])])
        out.append([vocab.index(Four_Gram[i][3])])
    x1 = we[index][:]
    x1 = x1.reshape(len(F), 48)
    y1 = we[out][:]
    y1 = y1.reshape(len(F), 16)
    if tanh == True:
        a1 = np.dot(x1,W1) + bias1
        a2 = np.tanh(a1)
        a3 = np.dot(a2,W2) + bias2
                    #softmax
        e = np.exp(a3) 
        s = np.sum(e,axis =1)
        s = s.reshape(len(F),1)
        Softmax = e/s
    else:
        a1 = np.dot(x1,W1) + bias1
        a2 = np.dot(a1,W2) + bias2
                #softmax
        e = np.exp(a2) 
        s = np.sum(e,axis =1)
        s = s.reshape(len(F),1)
        Softmax = e/s
        
    y_hot = np.zeros([len(F),8000])
    for i in range(len(F)):
        y_hot[i][out[i]] = 1
    loss = -y_hot * np.log(Softmax)    
    L = np.sum(loss)/len(F)
    
    return L

def predict(WE,vocab,W1,W2,b1,b2,Words):
    
    Word_Embed =np.array( [WE[vocab.index(Words[0])], WE[vocab.index(Words[1])], WE[vocab.index(Words[2])]  ])
    Word_Embed = Word_Embed.reshape(1,48)
    a1 = np.dot(Word_Embed,W1) + b1
    a2 = np.tanh(a1)
    a3 = np.dot(a2,W2) + b2
            #softmax
    e = np.exp(a3) 
    s = np.sum(e,axis =1)
    
    softmax = e/s
    ind = np.argmax(softmax)
    prediction = vocab[ind]
    return prediction

def distance(word,WE,vocab):
    we1 = WE[vocab.index(word[0])]
    we2 = WE[vocab.index(word[1])]
    dist =np.sqrt(np.sum( (we1-we2) ** 2))
    return dist
    
Four_Gram,diction, Gram_Val = preprocessing()
f = open('preproc.pckl','wb')
pickle.dump([Four_Gram,diction,Gram_Val],f,protocol=2)
f.close()
f = open('/Users/dhanashreebalaram/Downloads/HW3_DL/preproc.pckl','rb')
[Four_Gram,diction,Gram_Val] = pickle.load(f)
f.close()
print ('Done with Preprocessing') 

vocab = [item[0] for item in diction]
v =  [list(element) for element in diction]
batch_size = 64

WE, W1,W2,b1,b2 = initialize()

alpha = 0.01
Divby = len(Four_Gram)/batch_size
epochs = 20
flag = 1
Train_Loss = []
Val_Loss = []
Perp =[]


for ep in range(epochs):
    
    for i in range(0,len(Four_Gram), batch_size):
        idx = []
        output = []
        g = Four_Gram[i:i+batch_size][:]
        for j in range(len(g)):
            idx.append([vocab.index(g[j][0]),vocab.index(g[j][1]),vocab.index(g[j][2])])
            output.append([vocab.index(g[j][3])])
        X = WE[idx][:]
        
        if X.shape[0]==batch_size:
            
            X = X.reshape(batch_size, 48)
            X_update = X
            Y = WE[output][:]
            Y = Y.reshape(batch_size,16)
            #forward
            if tanh==False:
                a1 = np.dot(X,W1) + b1
                a2 = np.dot(a1,W2) + b2
                #softmax
                e = np.exp(a2) 
                s = np.sum(e,axis =1)
                s = s.reshape(batch_size,1)
                Softmax = e/s
              
                #backward
                Y_hot = np.zeros([batch_size,8000])
                for i in range(64):
                    Y_hot[i][output[i]] = 1
                    
                #Value = np.dot(Y,WE.T)
                deltaL = Softmax - Y_hot
                deltaW2 = np.dot(a1.T ,deltaL)/batch_size
                deltab2 = deltaL
                t = np.dot(deltaL , W2.T) # 1-y^2
                deltaW1 = np.dot( X.T, t)/batch_size
                deltab1 = np.dot(deltaL ,W2.T)
                deltaX = np.dot(np.dot(deltaL ,W2.T) ,W1.T)
            else:
                a1 = np.dot(X,W1) + b1
                a2 = np.tanh(a1)
                a3 = np.dot(a2,W2) + b2
                #softmax
                e = np.exp(a3) 
                s = np.sum(e,axis =1)
                s = s.reshape(batch_size,1)
                Softmax = e/s
                #backward
                Y_hot = np.zeros([batch_size,8000])
                for i in range(64):
                    Y_hot[i][output[i]] = 1
                    
                #Value = np.dot(Y,WE.T)
                deltaL = Softmax - Y_hot
                deltaW2 = np.dot(a2.T ,deltaL)/batch_size
                deltab2 = deltaL
                t = np.dot(deltaL , W2.T) * (1-(a2*a2)) # 1-y^2
                deltaW1 = np.dot( X.T, t)/batch_size
                deltab1 = t
                deltaX = np.dot(t ,W1.T)
            
            updateb1 = np.sum(deltab1,axis = 0)/batch_size
            updateb2 = np.sum(deltab2,axis = 0)/batch_size
            updateb1 = updateb1.reshape(1,hidden)
            updateb2 = updateb2.reshape(1,8000)
            
            #update
            W1 = W1-  alpha*(deltaW1)
            W2 = W2-  alpha*(deltaW2)
            b1 = b1-  alpha*updateb1
            b2 = b2-  alpha*updateb2
            X_update = X_update - alpha*deltaX
            
            X_update = X_update.reshape(batch_size,3,16)
            WE[idx][:] = X_update
            
            
    print('-----------------------------------------------------------')
    print ('Epoch: ', ep+1)
    TrainLoss = CalcL(WE,Four_Gram,W1,W2,b1,b2)
    Train_Loss.append(TrainLoss)
    ValLoss =  CalcL(WE,Gram_Val,W1,W2,b1,b2)
    P = np.exp(ValLoss)
    Val_Loss.append(ValLoss)
    Perp.append(P)
    print('Train Loss is :',TrainLoss,' Validation Loss is: ', ValLoss)
    print('Perplexity: ', P)
    
p1, = plt.plot(range(epochs),Train_Loss,'b')
p2, = plt.plot(range(epochs),Val_Loss,'g')

plt.ylabel('Cross Entropy Loss')
plt.xlabel('Epochs')
plt.title(' CROSS ENTROPY  PLOTS')

plt.legend([p1,p2],['Training Loss','Validation Loss'])
plt.show()  

p3, = plt.plot(range(epochs),Perp,'r')
plt.ylabel('Perplexity')
plt.xlabel('Epochs')
plt.title(' PERPLEXITY PLOT')
plt.legend([p3],['Validation Perplexity'])
plt.show() 
#        
#        
#        
# 

if tanh==False:           
    f = open('Language.pckl','wb')
    pickle.dump([WE,W1,W2,b1,b2,Train_Loss,Val_Loss,Perp],f,protocol=2)
    f.close()   
else:
    f = open('/Users/dhanashreebalaram/Downloads/HW3_DL/Tanh.pckl','wb')
    pickle.dump([WE,W1,W2,b1,b2,Train_Loss,Val_Loss,Perp],f,protocol=2)
    f.close()  
    
#        
# 
#f = open('Language.pckl','rb')
#[WE,W1,W2,b1,b2,Train_Loss,Val_Loss,Perp] = pickle.load(f)
#f.close()    


###############################################################################
##### ANALYSIS -- INPUTTING WORDS AND GETTING THE FOURTH WORD   ###############
###############################################################################       
if Analysis == True:
    f = open('/Users/dhanashreebalaram/Downloads/HW3_DL/Tanh.pckl','rb')
    [WE,W1,W2,b1,b2,Train_Loss,Val_Loss,Perp] = pickle.load(f)
    f.close() 
    Words = ['city', 'of', 'new']
    for i in range(10):
        pred = predict(WE,vocab,W1,W2,b1,b2,Words)
        print (pred)
        if pred != 'END':
            Words = [Words[1],Words[2], pred]
        else:
            break
    
    Words = ['review','tobacco']
    d = distance(Words,WE,vocab)
    print ('The Euclidean distance is ', d)    
    
###############################################################################
#########################     VISUALIZING     ################################# 
###############################################################################   

if Visualize == True:
    f = open('Visualizing.pckl','rb')
    [WE,W1,W2,b1,b2] = pickle.load(f)
    f.close() 
    
    a= WE[range(500),0]
    b= WE[range(500),1]
    v = vocab[0:499]
    fig, ax = plt.subplots(figsize=(30,10))
    ax.scatter(a,b)
    
    for i, txt in enumerate(v):
        ax.annotate(txt, (a[i],b[i])) 
        
    fig.savefig('Vis.png')  
        