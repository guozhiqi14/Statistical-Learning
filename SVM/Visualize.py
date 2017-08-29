# encoding=utf-8
import matplotlib.pyplot as plt
from generate_dataset import *
%matplotlib inline
plt.style.use('ggplot') 

train_features, train_labels, test_features, test_labels = generate_dataset(2000,visualization=False)

aaa = np.asarray(train_features)
fig = plt.figure(figsize=(10,8))

#Visualize in two-dimension
plt.scatter(aaa[:,0],aaa[:,1])


#Visualize in three-dimension
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(12,10)) 
ax = plt.axes(projection='3d')
#ax.plot3D(xline, yline, zline, 'gray')
ax.scatter3D(aaa[:,0], aaa[:,1], train_labels, cmap='Greens');