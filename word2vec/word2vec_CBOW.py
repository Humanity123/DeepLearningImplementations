import os
import numpy as np
import tensorflow as tf
import collections
import random
import math
import sklearn


dataIndex    = 0
volcabSize   = 50000
batchSize    = 100
windowSize   = 2
embeddingSize= 128 
negSamples   = 64
inFile       = "../data/text8"  

def getWords(inFile):
	with open(inFile,"rb") as dataFile:
		data = tf.compat.as_str(dataFile.read()).split()
	return data


def generateData(words):
	freq = [['UNK',0]]
	freq.extend(collections.Counter(words).most_common(volcabSize-1))
	dictionary = dict()
	for item in freq:
		dictionary[item[0]] = len(dictionary)
	data = list()
	print len(words)
	for word in words:
		if word in dictionary:
			data.append(dictionary[word])
		else:
			freq[0][1] += 1
			data.append(dictionary['UNK'])
	reverseDict = dict(zip(dictionary.values(),dictionary.keys()))
	return dictionary, reverseDict, freq, data

def startFresh(span, buffer):
	dataIndex = 0
	buffer.clear()
	for _ in range(span):
		buffer.append(data[dataIndex])
		dataIndex = (dataIndex + 1)%len(data)
 

def generateBatch(batchSize, windowSize):
	global dataIndex
	batch = np.ndarray(shape = [batchSize,2*windowSize], dtype = np.int32)
	label = np.ndarray(shape = [batchSize,1], dtype = np.int32)
	span = 2*windowSize + 1
	buffer = collections.deque(maxlen = span)
	for _ in range(span):
		if len(buffer) != 0 and dataIndex == 0:
			startFresh(span, buffer)
			break
		buffer.append(data[dataIndex])
		
		dataIndex = (dataIndex + 1)%len(data)
	for i in range(batchSize):
		windowIterator = 0
		for j in range(len(buffer)):
			if j != windowSize:
				batch[i][windowIterator] = buffer[j]
				windowIterator += 1

		label[i][0] = buffer[windowSize]
		if len(buffer) != 0 and dataIndex == 0:
			startFresh(span, buffer)
		else :
			buffer.append(data[dataIndex])
			dataIndex = (dataIndex + 1)%len(data)
	return batch, label

graph = tf.Graph()

with graph.as_default():

	train_input = tf.placeholder(tf.int32, shape = [batchSize, 2*windowSize])
	train_label = tf.placeholder(tf.int32, shape = [batchSize,1])

	embeddings  = tf.Variable(tf.random_uniform([volcabSize, embeddingSize], -1.0, 1.0))
	embed       = tf.nn.embedding_lookup(embeddings, train_input)
	train_embed_input = tf.div(tf.reduce_sum(embed,1), 2*windowSize)

	weights     = tf.Variable(tf.truncated_normal([volcabSize, embeddingSize],stddev=1.0 / math.sqrt(embeddingSize)))
	bias        = tf.Variable(tf.zeros(shape = [volcabSize]))

	nce_loss    = tf.reduce_mean(tf.nn.nce_loss(weights, bias, train_label, train_embed_input, negSamples, volcabSize)) 

	optimiser   = tf.train.GradientDescentOptimizer(1.0).minimize(nce_loss)	

	init        = tf.global_variables_initializer()

	norm 		= tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  	normalized_embeddings = embeddings / norm


epochs = 10000


dictionary, reverseDict, freq, data = generateData(getWords(inFile))
print "Data Generated"

with tf.Session(graph = graph) as sess:
	init.run()
	echo_loss = 0

	for epoch in range(1,epochs+1):
		batch, label = generateBatch(batchSize, windowSize)
		_, loss = sess.run([optimiser,nce_loss], feed_dict = {train_input : batch, train_label : label })

		echo_loss += loss
		if epoch % 100 == 0:
			print "Loss for 100 epochs is - ",echo_loss/100
			echo_loss = 0

	result = normalized_embeddings.eval()

def plot_with_labels(low_dim_embs, labels, filename='tsne_CBOW.png'):
  assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
  plt.figure(figsize=(18, 18))  #in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i,:]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

  plt.savefig(filename)

try:
  from sklearn.manifold import TSNE
  import matplotlib.pyplot as plt

  tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
  plot_only = 500
  low_dim_embs = tsne.fit_transform(result[:plot_only,:])
  labels = [reverseDict[i] for i in xrange(plot_only)]
  plot_with_labels(low_dim_embs, labels)

except ImportError:
  print("Please install sklearn and matplotlib to visualize embeddings.")















