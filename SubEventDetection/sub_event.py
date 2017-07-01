from math import *
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import csv
import sklearn 
import numpy as np
import datetime
import re
import similarity_score as ss

def cluster_date_similarity_score(cluster, point, feature_index, doc_list):
	''' return the similarity score of a cluster and a point with date as feature'''
	return ss.date_similarity_score(cluster.centroid, point[feature_index])

def cluster_location_similarity_score(cluster, point, feature_index, doc_list):
	''' return the similarity score of a cluster and a point with location as feature'''
	return ss.location_similarity_score(cluster.centroid, point[feature_index])

def cluster_text_similarity_score(cluster, point, feature_index, doc_list):
	''' return the similarity score of a cluster and a point with text as feature'''
	return ss.cosine_similarity_score(cluster.centroid, point[feature_index])

def cluster_grouping_similarity_score(cluster, point, feature_index_list, doc_list):
	''' returns the similarity score based on their clustering similarity based on other features in feature_index_list'''
	total_score = 0
	for feature_index in feature_index_list:
		total_score += sum([point[feature_index]['cluster_weight'] for doc_index in cluster.doc_list if doc_list[doc_index][feature_index]['cluster_index'] == point[feature_index]['cluster_index']])
	return total_score / float(len(cluster.doc_list))


class Cluster:
	'''class for a cluster'''
	def __init__(self, feature, feature_index, similarity_metric, MBS_coeff, doc, doc_index, use_MBS = 'True', threshold_func=None, use_centroid=True):
		''' feature_index is the index of the feature in the datapoints that would be processes
		by default MBS algo is used otherwise threshold function needs to be provided'''
		self.doc_list = [doc_index]
		self.feature = feature
		self.feature_index = feature_index
		self.similarity_metric = similarity_metric
		self.MBS = 1
		self.MBS_prev = 0
		self.MBS_coeff = MBS_coeff
		self.use_MBS = use_MBS
		self.threshold_func = threshold_func
		self.use_centroid = use_centroid
		if use_centroid == True:
			self.centroid = doc[feature_index]

	def similarity_score(self, doc, doc_list):
		return self.similarity_metric(self, doc, self.feature_index, doc_list)

	def get_threshold(self):
		if self.use_MBS:
			return self.MBS*self.MBS_coeff
		else:
			return self.threshold_func()

	def average_similarity(self, doc_list):
		return sum([self.similarity_metric(self, doc_list[index], self.feature_index, doc_list) for index in self.doc_list]) / float(len(self.doc_list))

	def update_MBS(self, doc_list, doc_index):
		if len(self.doc_list) <= 25 :
			self.MBS = self.average_similarity(doc_list)
		elif self.MBS_prev >= 20:
			self.MBS_prev = 0
			self.MBS = self.average_similarity(doc_list)
		else: 
			self.MBS_prev += 1
			self.MBS = (self.MBS * (len(self.doc_list) - 1) + self.similarity_metric(self, doc_list[doc_index], self.feature_index, doc_list )) / float(len(self.doc_list))

	def add_point(self, doc_list, doc_index):
		self.doc_list.append(doc_index)
		if self.use_centroid == True:
			self.centroid = sum([ doc_list[index][self.feature_index] for index in self.doc_list])/len(self.doc_list)
		if self.use_MBS:
			self.update_MBS(doc_list, doc_index)

	def print_cluster(self, doc_list, feature_index_list):
		print "[",
		for doc_index in self.doc_list[:5]:
			print "[",
			for feature_index in feature_index_list:
				print doc_list[doc_index][feature_index],",",
			print "]"
		print "]"

class ClusterSolution:
	'''class for cluster solution for a feature'''
	def __init__(self, feature, feature_index, similarity_metric, MBS_coeff=0, use_MBS = 'True', threshold_func=None, use_centroid=True):
		''' feature_index is the index of the feature in the datapoints that would be processes'''
		self.cluster_list = []
		self.doc_to_cluster_mapping = {}
		self.feature = feature
		self.feature_index = feature_index
		self.similarity_metric = similarity_metric
		self.MBS_coeff = MBS_coeff
		self.use_MBS = use_MBS
		self.threshold_func = threshold_func
		self.use_centroid = use_centroid

	def get_cluster_index(self, doc_index):
		try:
			return self.doc_to_cluster_mapping[doc_index]
		except:
			print "doc_index not encountered yet"
			return -1

	def add_point(self, doc_list, doc_index):
		if len(self.cluster_list) == 0:
			print "***Adding a new cluster***"
			print "cluster_index: ",len(self.cluster_list)
			print "doc_index: ",doc_index
			self.cluster_list.append(Cluster(self.feature, self.feature_index, self.similarity_metric, self.MBS_coeff, doc_list[doc_index], doc_index, self.use_MBS, self.threshold_func, self.use_centroid))
			self.doc_to_cluster_mapping[doc_index] = 0
			return 
		
		max_similarity_score_index = 0
		max_similarity_score = self.cluster_list[0].similarity_score(doc_list[doc_index], doc_list)
		for cluster_index, cluster in enumerate(self.cluster_list):
			cluster_similarity_score = cluster.similarity_score(doc_list[doc_index], doc_list)
			if cluster_similarity_score > max_similarity_score:
				max_similarity_score = cluster_similarity_score
				max_similarity_score_index = cluster_index

		if max_similarity_score >= self.cluster_list[max_similarity_score_index].get_threshold():
			self.cluster_list[max_similarity_score_index].add_point(doc_list, doc_index)
			self.doc_to_cluster_mapping[doc_index] = max_similarity_score_index
		else :
			print max_similarity_score, self.cluster_list[max_similarity_score_index].get_threshold()
			print "***Adding a new cluster*** "
			print "cluster_index: ",len(self.cluster_list)
			print "doc_index: ",doc_index
			self.cluster_list.append(Cluster(self.feature, self.feature_index, self.similarity_metric, self.MBS_coeff, doc_list[doc_index], doc_index, self.use_MBS, self.threshold_func, self.use_centroid))
			self.doc_to_cluster_mapping[doc_index] = len(self.cluster_list) - 1

	def reset(self):
		'''removes existing solution due to previous points'''
		self.cluster_list = []
		self.doc_to_cluster_mapping = {}
		return

	def make_cluster_solution(self, doc_list):
		''' creates a cluster solution from the given list of points''' 
		self.reset()
		for doc_index in range(len(doc_list)):
			self.add_point(doc_list, doc_index)
		return 

	def print_cluster_solution(self, doc_list, feature_index_list):
		''' prints the cluster solution '''
		print "PRINTING CLUSTER SOLUTION\n"
		for cluster_index, cluster in enumerate(self.cluster_list):
			print "CLUSTER NO - ",cluster_index
			cluster.print_cluster(doc_list, feature_index_list)
			print "\n"


def threshold_for_cluster():
	''' threshold for making clusters after making the clusters with local features'''
	''' Trying Majority similarity for Threshold'''
	return 0.5

def get_tfidf_scores(list_of_tweets):
	#list_of_tweets = [ doc[0] for doc in list_of_docs ]
	vectorizer = TfidfVectorizer(stop_words = 'english')
	tfidf_score = np.array(vectorizer.fit_transform(list_of_tweets).todense())
	feature_names = vectorizer.get_feature_names()
	#return feature_names
	tfidf_score_list = []
	for index in range(tfidf_score.shape[0]):
		tfidf_score_list.append(tfidf_score[index])
	return tfidf_score_list

def clean_doc(doc):
	'''cleans the doc by lemmatsing the verbs and converting to lower case, 
	handles have been left as they contain usefull information of the person addressed'''
	tknzr = TweetTokenizer(reduce_len=True)
	lmtzr = WordNetLemmatizer()
	tokens = tknzr.tokenize(doc.decode('utf-8'))
	cleaned_tokens = [lmtzr.lemmatize(token.lower(), pos='v') for token in tokens]
	cleaned_doc=""
	for token in cleaned_tokens: 
		cleaned_doc += token + " "
	return cleaned_doc.strip()

def create_docs(tweets):
	'''Given File Containing Tweets returns lists of dics with each doc having the tweet text and days passed since epoch and tfidf scores'''
	with open(tweets,'rb') as tweet_csv:
		reader = csv.DictReader(tweet_csv)
		tweet_text = []
		tweet_seconds_since_epoch = []
		tweet_text_for_tfidf = []
		for tweet in reader:
			text = clean_doc(tweet['Tweet Text'])
			text_for_tfidf = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",text).split())
			seconds_since_epoch = ( datetime.datetime.strptime(tweet['Tweet Created At'],"%Y-%m-%d %H:%M:%S") - datetime.datetime(1970,1,1) ).total_seconds()
			tweet_text.append(text)
			tweet_seconds_since_epoch.append(seconds_since_epoch)
			tweet_text_for_tfidf.append(text_for_tfidf)
	tweet_tfidf_score_list = get_tfidf_scores(tweet_text_for_tfidf)
	return zip(tweet_text, tweet_seconds_since_epoch, tweet_tfidf_score_list)

def create_docs_with_cluster_information(docs, cluster_solution_list, cluster_weight_list):
	''' create new docs by appending the cluster index, cluster weight information for each feature to each point'''
	if len(docs) ==0 :
		print "docs are empty"
		return
	all_lists = [ [ doc[index] for doc in docs] for index in range(len(docs[0])) ]
	for cluster_solution, cluster_weight in  zip(cluster_solution_list, cluster_weight_list):
		index_in_all_lists = len(all_lists)
		all_lists.append([])
		for doc_index, doc in enumerate(docs):
			all_lists[index_in_all_lists].append( {"cluster_index":cluster_solution.get_cluster_index(doc_index) , "cluster_weight":cluster_weight} )

	return zip(*all_lists)

def create_clusters():
	docs = create_docs('tweets_nepal_earthquake.csv')
	cluster_solution_by_tweet = ClusterSolution('text', 2, cluster_text_similarity_score, 0.1)
	cluster_solution_by_date = ClusterSolution('time', 1, cluster_date_similarity_score, 0.8)
	cluster_solution_by_tweet.make_cluster_solution(docs)
	cluster_solution_by_date.make_cluster_solution(docs)
	docs_with_cluster_information = create_docs_with_cluster_information(docs, [cluster_solution_by_tweet, cluster_solution_by_date], [0.8, 0.2])
	cluster_solution_by_local_clustering = ClusterSolution('local_clusters', [3,4], cluster_grouping_similarity_score, use_MBS=False, threshold_func=threshold_for_cluster, use_centroid=False)
	cluster_solution_by_local_clustering.make_cluster_solution(docs_with_cluster_information)
	print "NUMBER OF CLUSTERS BY TEXT - ", len(cluster_solution_by_tweet.cluster_list)
	print "NUMBER OF CLUSTERS BY DATE - ", len(cluster_solution_by_date.cluster_list)
	print "NUMBER OF FINAL CLUSTERS   - ", len(cluster_solution_by_local_clustering.cluster_list)
	cluster_solution_by_local_clustering.print_cluster_solution(docs_with_cluster_information, [0])
	

def main():
	return create_clusters()
if __name__ == '__main__':
    main()






