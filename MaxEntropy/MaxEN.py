import pandas as pd
import numpy as np

import time
import math
import random

from collections import defaultdict

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score


#read data
raw_data = pd.read_csv('train.csv',header=0)
raw_data = raw_data[:1000]
data = raw_data.values

imgs = data[:,1:]
labels = data[:,0]

train_set, test_set, train_labels, test_labels = train_test_split(imgs,labels,test_size=0.33,random_state=23333)

def rebuild_features(features):
    new_features=[]
    
    for feature in features:
        new_feature = []
        
        for i,j in enumerate(feature):
            new_feature.append(str(i)+'_'+str(j))
        
        new_features.append(new_feature)
        
    return new_features


train_set = rebuild_features(train_set)
test_set = rebuild_features(test_set)


class MaxEntropy(object):
    
    def init_params(self,X,Y):
        self.X = X
        self.Y = set()
        
        self.cal_Pxy_Px(X,Y)
        
        self.N = len(X)
        self.n = len(self.Pxy) 
        self.M = 1000.0 
        
        self.build_dict()
        self.cal_EPxy() 
    
    
    def cal_Pxy_Px(self,X,Y):
        '''
        Count empirical appearance of each feature and feature-value pair
        '''
        self.Pxy = defaultdict(int) 
        self.Px  = defaultdict(int)
        
        for i in range(len(X)):
            x_,y = X[i],Y[i]
            self.Y.add(y)
            
            for x in x_:
                self.Pxy[(x,y)] += 1
                self.Px[x] += 1
        
    def build_dict(self):
        self.id2xy = {}
        self.xy2id = {}
        
        for i, (x,y) in enumerate(self.Pxy):
            self.id2xy[i] = (x,y)
            self.xy2id[(x,y)] = i
    
    def cal_EPxy(self):
        
        self.EPxy = defaultdict(float)
        
        for id in range(self.n):
            (x,y) = self.id2xy[id]
            self.EPxy[id] = float(self.Pxy[(x,y)]) / float(self.N) #why id here 
    
    
    def fxy(self,x,y):
        return (x,y) in self.xy2id
    
    def cal_pyx(self,X,y):
        '''
        Helper for cal_probability()
        '''
        result = 0.0
        for x in X:
            if self.fxy(x,y):
                id = self.xy2id[(x,y)]
                result += self.w[id]  
        
        return (np.exp(result),y) 
    
    def cal_probability(self,X):
      
        Pyxs = [(self.cal_pyx(X, y)) for y in self.Y]
        Z = np.sum([prob for prob,y in Pyxs]) 
        return [(prob/Z, y) for prob,y in Pyxs]        
        
    def cal_EPx(self):
        
        self.EPx = [0.0 for i in range(self.n)]
    
        for i,X in enumerate(self.X):
            Pyxs = self.cal_probability(X) 
            
            for x in X:
                for Pyx, y in Pyxs:
                    if self.fxy(x,y):
                        id = self.xy2id[(x,y)]
                        
                        self.EPx[id] += Pyx*(1.0/self.N) ###why 1/length
                        #self.EPx[id] += Pyx*(self.Px[x]/self.n)
        
    def train(self,X,Y):
        '''
        Train MaxEn model
        '''
        self.init_params(X,Y)
        self.w = [0.0 for i in range(self.n)]
        self.w = np.array(self.w)
        
        #set default iteration to 1000(since model is time-consuming) 
        max_iteration = 10 
        
        for iterate in range(max_iteration):
            sigmas = []
            self.cal_EPx()
            
            for i in range(self.n):
                sigma = 1 / self.M * np.log(self.EPxy[i]/self.EPx[i])
                sigmas.append(sigma)
            
            sigmas = np.array(sigmas)
            self.w += sigmas
            #print sigmas
    
    def predict(self,testset):
        results = []
        for test in testset:
            result = self.cal_probability(test)
            
            results.append((max(result, key=lambda x: x[0])[1]))
        return results




met = MaxEntropy()
met.train(train_set, train_labels)

time_3 = time.time()
print 'training cost ', time_3 - time_2, ' second', '\n'

print 'Start predicting'
test_predict = met.predict(test_set)
time_4 = time.time()
print 'predicting cost ', time_4 - time_3, ' second', '\n'

score = accuracy_score(test_labels, test_predict)
print score