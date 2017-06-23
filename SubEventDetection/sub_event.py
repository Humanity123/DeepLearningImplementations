from math import *
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import csv
import sklearn 
import numpy as np
import datetime

def tf_idf( doc, term, list_of_docs):
	num_docs_containing_term = 0.0
	for document in list_of_docs:
		if term in document:
			num_docs_containing_term += 1

	if term in doc :
		freq_term_in_doc = doc[term]
	else :
		freq_term_in_doc = 0

	words_in_doc = 1.0 * sum(doc.values())
	return (freq_term_in_doc / words_in_doc) * tf.log( len(list_of_docs) / num_docs_containing_term)

def date_similarity_score(date1, date2):
	''' if the difference in days is more than one month apart it is taken as 0'''
	diff_in_days = abs(date1-date2)
	seperation_limit_in_days = 30.0
	if diff_in_days > seperation_limit_in_days:
		return 0.0
	return 1- ( diff_in_days / seperation_limit_in_days )

def location_similarity_score(location1, location2):
	''' similarity between two locations using the widely accepted Haversine Distance'''
	longitude_1, latitude_1, longitude_2, latitude_2 = map(radians, location1[0], location1[1], location2[0], location2[1])
	X1 = ( sin((latitude_2-latitude_1)/2.0) )**2
	X2 = cos(latitude_1)*cos(latitude_2)*sin( ((longitude_2-longitude_1)/2.0)**2 )
	d = 2*asin( (X1+X2)**0.5 )
	return 1-d

def cosine_similarity_score(vector1, vector2):
	'''similarity between two tfidf score vectors of documents'''
	return sklearn.metrics.pairwise.cosine_similarity( np.array(vector1).reshape(1,-1), np.array(vector2).reshape(1,-1) )[0][0]


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
		return self.similarity_metric(self.centroid, doc[self.feature_index])

	def get_threshold():
		return self.MBS*self.MBS_coeff

	def average_similarity(self, doc_list):
		return sum([self.similarity_metric(self.centroid, doc_list[index][self.feature_index]) for index in self.doc_list]) / float(len(self.doc_list))

	def update_MBS(self, doc_list, doc_index):
		if len(self.doc_list <= 25 ):
			self.MBS = average_similarity(doc_list)
		elif self.MBS_prev >= 20:
			self.MBS_prev = 0
			self.MBS = average_similarity(doc_list)
		else: 
			self.MBS = (self.MBS * (len(self.doc_list) - 1) + self.similarity_metric(self.centroid, doc_list[doc_index][self.feature_index])) / float(len(self.doc_list))


	def add_point(self, doc_list, doc_index):
		self.doc_list.append(doc_index)
		self.centroid = sum([ doc_list[index][self.feature_index] for index in self.doc_list])/len(self.doc_list)
		self.update_MBS(doc_list, doc_index)






class ClusterSolution:
	'''class for cluster solution for a feature'''
	def __init__(self, feature, feature_index, similarity_metric):
		''' feature_index is the index of the feature in the datapoints that would be processes'''
		self.cluster_list = []
		self.feature = feature
		self.feature_index = feature_index
		self.similarity_metric = similarity_metric

	


def get_tfidf_scores(list_of_docs):
	list_of_tweets = [ doc[0] for doc in list_of_docs ]
	vectorizer = TfidfVectorizer(stop_words = 'english')
	tfidf_score = np.array(vectorizer.fit_transform(list_of_tweets).todense())
	tfidf_score_list = []
	for index in range(tfidf_score.shape[0]):
		tfidf_score_list.append(tfidf_score[index])
	return tfidf_score_list

def clean_doc(doc):
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






