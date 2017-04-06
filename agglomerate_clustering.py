#encoding=utf-8
'''
做舆情检测，输入的句子大概只有10000句左右，然后要找出其中的热点信息
这里先用1-gram模型来编码句子，再降维到256维，然后就可以得到两两句子之间的相似度
算法： 
	计算一个clust和其他所有之间所有点的平均距离simi，
		simi小于epsilon
			clust里面元素太少，直接删去，里面元素很多，作为一个离散的topics提取出来；
		simi大于epsilon
			取相似度最高的那个clust，和原来的clust融合；

'''
import sys, os 


class Text_clustering:
	def __init__(self, simi_matrix, epsilon):
		self.simi_matrix = simi_matrix
		self.epsilon = epsilon

	def agglomerate_clustering(self):
	
		N_points = range(self.simi_matrix.shape[0])
		clusters = {}
		for i in range(len(N_points)):
			clusters[i] = [N_points[i]]

		topics = [] 
		while len(clusters)>=1:
			key = clusters.keys()
			k = key[0]
			simi_max = 0
			for s in key[1:]:
				dist = self.group_simi(clusters[k], clusters[s])
				if dist > simi_max:
					simi_max = dist 
					max_s = s

			if simi_max < self.epsilon:
				if len(clusters[k]) >= 20:
					topics.append(clusters[k])
				del clusters[k]
			else:
				clusters[max_s] = clusters[max_s] + clusters[k]
				del clusters[k]

			print len(clusters)

		key_sent = []	
		for clust in topics:
			z = self.group_to_one(clust)
			key_sent.append((z, len(clust)*1.0/len(N_points)))

		return key_sent

	def group_simi(self, clust1, clust2):
		z, count = 0.0, 0  
		for i in clust1:
			for j in clust2:
				z += self.simi_matrix[i][j]
				count += 1 
		return z/count 

	def group_to_one(self, clust):
		simi_max = 0.0
		for i in clust:
			dis = self.group_simi([i], clust)
			if dis > simi_max:
				z = i 
				simi_max = dis 
		return z
