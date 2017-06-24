from math import *
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import csv
import sklearn 
import numpy as np
import datetime
import similarity_score as ss

def cluster_date_similarity_score(cluster, point, feature_index):
	''' return the similarity score of a cluster and a point with date as feature'''
	return ss.date_similarity_score(cluster.centroid, point[feature_index])

def cluster_location_similarity_score(cluster, point, feature_index):
	''' return the similarity score of a cluster and a point with location as feature'''
	return ss.location_similarity_score(cluster.centroid, point[feature_index])

def cluster_text_similarity_score(cluster, point, feature_index):
	''' return the similarity score of a cluster and a point with text as feature'''
	return ss.cosine_similarity_score(cluster.centroid, point[feature_index])

class Cluster:
	'''class for a cluster'''
	def __init__(self, feature, feature_index, similarity_metric, MBS_coeff, doc, doc_index):
		''' feature_index is the index of the feature in the datapoints that would be processes'''
		self.doc_list = [doc_index]
		self.feature = feature
		self.feature_index = feature_index
		self.similarity_metric = similarity_metric
		self.MBS = 1
		self.MBS_prev = 0
		self.MBS_coeff = MBS_coeff
		self.centroid = doc[feature_index]

	def similarity_score(self, doc):
		return self.similarity_metric(self, doc[self.feature_index])

	def get_threshold():
		return self.MBS*self.MBS_coeff

	def average_similarity(self, doc_list):
		return sum([self.similarity_metric(self, doc_list[index][self.feature_index]) for index in self.doc_list]) / float(len(self.doc_list))

	def update_MBS(self, doc_list, doc_index):
		if len(self.doc_list <= 25 ):
			self.MBS = average_similarity(doc_list)
		elif self.MBS_prev >= 20:
			self.MBS_prev = 0
			self.MBS = average_similarity(doc_list)
		else: 
			self.MBS = (self.MBS * (len(self.doc_list) - 1) + self.similarity_metric(self, doc_list[doc_index][self.feature_index])) / float(len(self.doc_list))


	def add_point(self, doc_list, doc_index):
		self.doc_list.append(doc_index)
		self.centroid = sum([ doc_list[index][self.feature_index] for index in self.doc_list])/len(self.doc_list)
		self.update_MBS(doc_list, doc_index)

class ClusterSolution:
	'''class for cluster solution for a feature'''
	def __init__(self, feature, feature_index, similarity_metric, MBS_coeff):
		''' feature_index is the index of the feature in the datapoints that would be processes'''
		self.cluster_list = []
		self.doc_to_cluster_mapping = {}
		self.feature = feature
		self.feature_index = feature_index
		self.similarity_metric = similarity_metric
		self.MBS_coeff = MBS_coeff

	def get_cluster_index(doc_index):
		try:
			return self.doc_to_cluster_mapping[cluster_index]
		except:
			print "doc_index not encountered yet"
			return -1

	def add_point(self, doc_list, doc_index):
		if len(cluster_list) == 0:
			cluster_list.append(Cluster(self.feature, self.feature_index, self>similarity_metric, self.MBS_coeff, doc_list[doc_index], doc_index))
			self.doc_to_cluster_mapping[doc_index] = 0
			return 
		
		max_similarity_score_index = 0
		max_similarity_score = cluster_list[0].similarity_score(doc_list[doc_index])
		for cluster_index, cluster in enumerate(cluster_list):
			cluster_similarity_score = cluster.similarity_score(doc_list[doc_index])
			if cluster_similarity_score > max_similarity:
				max_similarity = cluster_similarity_score
				max_similarity_score_index = cluster_index

		if max_similarity_score >= cluster_list[max_similarity_score_index].get_threshold:
			cluster_list[max_similarity_score_index].add_point(doc_list, doc_index)
			self.doc_to_cluster_mapping[doc_index] = max_similarity_score_index
		else :
			cluster_list.append(Cluster(self.feature, self.feature_index, self>similarity_metric, self.MBS_coeff, doc_list[doc_index], doc_index))
			self.doc_to_cluster_mapping[doc_index] = len(cluster_list) - 1

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










def get_tfidf_scores(list_of_docs):
	list_of_tweets = [ doc[0] for doc in list_of_docs ]
	vectorizer = TfidfVectorizer(stop_words = 'english')
	tfidf_score = np.array(vectorizer.fit_transform(list_of_tweets).todense())
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
		tweet_days_since_epoch = []
		for tweet in reader:
			text = clean_doc(tweet['Tweet Text'])
			days_since_epoch = ( datetime.datetime.strptime(tweet['Tweet Created At'],"%Y-%m-%d %H:%M:%S") - datetime.datetime(1970,1,1) ).days
			tweet_text.append(text)
			tweet_days_since_epoch.append(days_since_epoch)
	tweet_tfidf_score_list = get_tfidf_scores(tweet_text)
	return zip(tweet_text, tweet_days_since_epoch, tweet_tfidf_score_list)


def main():
	return 
if __name__ == '__main__':
    main()






