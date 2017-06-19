import tensorflow as tf
import numpy as np
from math import *

def tf_idf( doc, term, list_of_docs):
	num_docs_containing_term = 0.0
	for doc in list_of_docs:
		if term in doc:
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




