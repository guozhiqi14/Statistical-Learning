#encoding=utf-8

'''
先验概率： 由于先验概率分母都是N，因此不用除于N，直接用分子即可。 
条件概率： 条件概率公式如下图所示，我们得到概率后再乘以10000，将概率映射到[0,10000]中，
但是为防止出现概率值为0的情况，人为的加上1，使概率映射到[1,10001]中。

Data Source: 
MNIST data 
https://www.kaggle.com/c/digit-recognizer/data
'''

import pandas as pd
import numpy as np
import cv2
import random
import time

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score


def binaryzation(img):
    cv_img = img.astype(np.uint8)
    cv2.threshold(cv_img,50,1,cv2.THRESH_BINARY_INV,cv_img)
    return cv_img

def Train(train_features,train_labels):
	prior_prob = np.zeros(class_num)       #prior prob 先验概率
	conditional_prob = np.zeros((class_num,feature_len,2))  #conditional probability 条件概率

	for i in range(len(train_labels)):
		img = binaryzation(train_features[i])  # 图片二值化
		label = train_labels[i]

		prior_prob[label] += 1

		for j in range((feature_len)):
			conditional_prob[label][j][img[j]] += 1

	# 将概率归到[1.10001]
	for i in range(class_num):
		for j in range(feature_len):
			# 经过二值化后图像只有0，1两种取值
			pix_0 = conditional_prob[i][j][0]
			pix_1 = conditional_prob[i][j][1]

			# 计算0，1像素点对应的条件概率
			probalility_0 = (float(pix_0)/float(pix_0+pix_1))*1000000 + 1
			probalility_1 = (float(pix_1)/float(pix_0+pix_1))*1000000 + 1

			conditional_prob[i][j][0] = probalility_0
			conditional_prob[i][j][1] = probalility_1

	return prior_prob, conditional_prob


#计算概率
def cal_probability(img,label):
	probability = int(prior_prob[label])

	for i in range(len(img)):
		probability *= int(conditional_prob[label][i][img[i]])

	return probability

def Predict(test_features,prior_prob,conditional_prob):

	prediction = []

	for i in test_features:
		img = binaryzation(i)

		max_label = 0
		max_prob = 	cal_probability(img,0)

		for j in range(1,10):
			probability = cal_probability(img,j)

			if probability > max_prob: #妈个鸡狗屁bug看了我1个小时.....
				max_label = j
				max_prob = probability

		prediction.append(max_label)

	return np.array(prediction)






class_num = 10
feature_len = 784 #pixel

if __name__ == '__main__':

	time1 = time.time()
	#read  MNIST data 
	raw_data = pd.read_csv('/Users/guozhiqi-seven/Documents/Statistical Learning/NaiveBayes/train.csv')

	time2 = time.time()

	print('Reading data cost', time2-time1, 'second.','\n')


	#reading 1000 rows from dataset for example to speed up 
	reading_row = int(input('How many rows:'))
	raw_data = raw_data[:reading_row]

	data = raw_data.values
	imgs = data[:,1:]
	labels = data[:,0]

	train_features, test_features, train_labels, test_labels = train_test_split(imgs,labels,test_size=0.25)

	time3 = time.time()
	prior_prob, conditional_prob = Train(train_features,train_labels)
	prediction = Predict(test_features, prior_prob,conditional_prob)
	time4 = time.time()

	score = accuracy_score(test_labels,prediction)

	print('Predicting data cost', time4-time3, 'second.','\n')
	print("The accruacy socre is ", score)




