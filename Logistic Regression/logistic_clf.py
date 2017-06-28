# encoding=utf-8

import pandas as pd 
import numpy as np 
import time

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

'''
1.求导的是likelihood function, not cost function
2.注意由于梯度中包含指数操作，所以需要一个很小的学习率。
'''

class logistic_regression(object):

	def __init__(self,max_iteration=5000,learning_rate=0.00001):
		self.max_iteration = max_iteration
		self.learning_rate = learning_rate

	def predict_(self,x):
		wx = np.dot(self.w,x)

		p_1 = np.exp(wx)/(1+np.exp(wx))
		p_0 = 1/(1+np.exp(wx))

		if p_1 >= p_0:
			return 1
		else:
			return 0

	def train(self,features,labels):
		self.w = [0.0] * (len(features[0])+1)

		#SGD
		iteration = 0

		while iteration < self.max_iteration:
			index = np.random.choice(len(features),1)[0]
			x = features[index]
			x = np.append(x,1)  #x=(x1,x2,....,xn,1)^T
			y = labels[index]

			'''
			Caution:
			Reference to chp6 p79. 此处对log-likelihood function 求导，我们想要likelihood最大，等价于其负值最小
			所有gradient之后要带负号
			'''
			gradient = y*x - np.exp(np.dot(self.w,x))*x / (1+np.exp(np.dot(self.w,x)))

			self.w -= self.learning_rate * (-1*gradient)
			iteration += 1

	def predict(self,features):
		labels = []

		for feature in features:
			x = list(feature)
			x.append(1)

			labels.append(self.predict_(x))
			
		return labels


