import numpy as np
from collections import Counter

def majority_vote(labels):
	"""assumes that labels are ordered from nearest to farthest"""
	vote_counts = Counter(labels)
	winner, winner_count = vote_counts.most_common(1)[0]

	num_winners = len([count for count in vote_counts.values() if count == winner_count ])

	#unique winner
	if num_winners == 1:
		return winner
	#try without farthest
	else:
		return majority_vote(labels[:-1])

def distance(point, new_point):
	temp = 0

	for (i,j) in zip (point,new_point):
		temp += (i-j)**2
	return np.sqrt(temp)

def knn_classify(K,labeled_points,new_point):
	"""each labeled point should be a pair (point, label)"""

	#order distance from closest to farthest
	by_distance = sorted(labeled_points, key = lambda x: distance(x[0],new_point))

	# find the labels for the k closest
	k_closest_labels = [label for _,label in by_distance[:K]]
	#print(k_closest_labels)
	#return the majority label
	return majority_vote(k_closest_labels)