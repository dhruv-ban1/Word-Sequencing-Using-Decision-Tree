import numpy as np
import random
import sklearn
from sklearn.tree import DecisionTreeClassifier
# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py
# DO NOT CHANGE THE NAME OF THE METHOD my_fit or my_predict BELOW
# IT WILL BE INVOKED BY THE EVALUATION SCRIPT
# CHANGING THE NAME WILL CAUSE EVALUATION FAILURE
# You may define any new functions, variables, classes here
# For example, classes to create the Tree, Nodes etc


# class DecisionTree:
# 	def __int__( self , min_sample_split, max_depth, min_leaf_size ):
# 		self.root  = None
# 		self.words= None 
# 		self.min_sample_split = min_sample_split
# 		self.max_depth = max_depth
# 		self.min_leaf_size = min_leaf_size

	
# 	def fit(self , words):
# 		self.words = words 
# 		self.root = None(depth = 0 , parent = None)

# 		self.root.fit(all_words = self.words , my_words_idx = np.arrange(len(self.words)), min_leaf_size  = self.min_leaf_size, max_depth = self.max_depth)
	

# 	def predict(self , bg):
# 		node = self.root 

# 		# go to child based on decision criterian
# 		while not node.is_leaf:
# 			node = node.get_child(node.get_query in bg)
		

# 		guesses = []
# 		cnt = 0
# 		for i in node.my_words_idx:
# 			if cnt == 5:
# 				return guesses
# 			guesses = self.words[i]
# 			cnt += 1 
		
	



# class Node:
# 		def __init__( self, depth, parent ):
# 			self.depth = depth
# 			self.parent = parent
# 			self.all_words = None
# 			self.my_words_idx = None
# 			self.children = {}
# 			self.is_leaf = True
# 			self.query = None
# 			self.history = []
		
# 		def get_query(self):
# 			return self.query
		

# 		def get_child(self , response):

# 			if self.is_leaf:
# 				chile = self
# 			else:
# 				if response not in self.children:
# 					response =  list(self.children.keys())[0]
				
# 				child = self.children[response]
			
# 			return child 


# def split_to_length(words):
# 	length_category = {
# 		'bglen3' :[],
# 		'bglen4' :[],
# 		'bglen_gt_5' :[]
# 	}

# 	for word in words:
# 		if len(word) == 4 :
# 			length_category['bglen3'].append(word)
		
# 		elif len(word) == 5 :
# 			length_category['bglen4'].apppend(word)
		
# 		elif len(word) >5 :
# 			length_category['bglen_gt_5'].append(word)
	
# 	return length_category

# def categorise_by_bigrams(words):
# 	length_cat = split_to_length(words) #it is a dictionary 
# 	length_3_bg = length_cat['len4']
# 	length_4_bg = length_cat['len5']
# 	length_g_5_bg = length_cat['len_gt_5']

# 	dic_3_bg ={}
# 	dic_4_bg = {}
# 	dic_5_bg = {}

# 	for word in length_3_bg: 
# 		dic_3_bg[word] = genrate_bigrams(word , lim = 3)
	
# 	for word in length_4_bg: 
# 		dic_4_bg[word] = genrate_bigrams(word , lim = 4)
	
# 	for word in length_g_5_bg: 
# 		dic_5_bg[word] = genrate_bigrams(word , lim = 5)


def genrate_bigrams( word, lim = None ):
  # Get all bigrams
  bg = map( ''.join, list( zip( word, word[1:] ) ) )
  # Remove duplicates and sort them
  bg = sorted( set( bg ) )
  # Make them into an immutable tuple and retain only the first few
  return tuple( bg )[:lim]

def multi_hot(words):
	all_bigrams = []
	bg_dictionary = {}

	for word in words:
		bg_list = list(genrate_bigrams(word))
		bg_dictionary[word] = bg_list
		all_bigrams.extend(bg_list)

	total_bigrams = sorted(set(all_bigrams))
	total_len = len(total_bigrams)
	# for word in words:
	# 	total_bigrams = [word[i:i+2] for i in range(len(word)-1)]
	
	# total_bigrams = list(set(total_bigrams))
	length = len(words)
	global word_dictionary 
	word_dictionary = {}
	words_idx = []
	for i in range(length):
		word_dictionary[i] = words[i]
		words_idx.append(i)

	# bg_dictionary = {}
	# for word in words :
	# 	bg_list = list(genrate_bigrams(word))
	# 	bg_dictionary[word] = bg_list

	
	# total_len = len(total_bigrams)
	multi_hot_vector = []
	for word in words:
		vector = [0] * total_len
		bgs = bg_dictionary[word]

		for i , bigram in enumerate(total_bigrams):
			if bigram in bgs:
				vector[i] = 1
		multi_hot_vector.append(vector)
	
	return multi_hot_vector, total_bigrams, words_idx




################################
# Non Editable Region Starting #
################################
def my_fit( words ):
################################
#  Non Editable Region Ending  #
################################

	# Do not perform any file IO in your code
	# Use this method to train your model using the word list provided
	
	global total_bigrams , x_train 
	x_train , total_bigrams, words_idx  = multi_hot(words) #training set of words converted to bigram multihot vector 
	# length = len(words)
	# word_dictionary= {}
	# for i in range(length):
	# 	word_dictionary[i] = words[i]
	# 	words_idx.append(i)

	y_train = words_idx

	model = DecisionTreeClassifier(random_state=10,criterion='gini') 
	model.fit(x_train , y_train)
	
	return model	


################################
# Non Editable Region Starting #
################################
def my_predict( model, bigram_list ):
################################
#  Non Editable Region Ending  #
################################
	
	# Do not perform any file IO in your code
	# Use this method to predict on a test bigram_list
	# Ensure that you return a list even if making a single guess
	#converting to one multi hot vector
	hot_vector = [0]*len(total_bigrams)
	for i , bigram in enumerate(total_bigrams):
		if bigram in bigram_list:
			hot_vector[i] = 1
	

	# probabilities = model.predict_proba([hot_vector])[0]
	# top_indices = np.argsort(probabilities)[::-1][:5]
	guess_list = []
	guess_list_idx = model.predict([hot_vector])
	for idx in guess_list_idx:
		guess_list.append(word_dictionary[idx])

	# while i > 0 :
	# 	guess_list.append(' '.join([f"{item}" for item in model.predict([hot_vector] )]))
	# 	i = i-1
	return guess_list[:5] 		



# words = ["hello", "world", "example", "testing", "python", "program", "function", "variable", "apple", "banana"]
# model  = my_fit(words)

# # Example bigram list to predict
# bigram_list = genrate_bigrams("example")

# Predict top 5 words
# predicted_words  = my_predict(model, bigram_list)
# print(f"Predicted words for bigram list {bigram_list}: {predicted_words}")

# # Evaluate precision on the test set
# precision = 0
# for word in words:
# 	bigram_list = genrate_bigrams(word)
# 	predictions = my_predict(model, bigram_list)
# 	if word in predictions:
# 		precision += 1 / len(predictions)

# precision /= len(words)
# print(f"Precision: {precision:.6f}")
# print(predicted_size)
# # words = ['apple','applaud' , 'applying',  'appealing', 'beat' , 'beauty', 'beautify', 'beast' ]
# # model = my_fit(words)
# # test_bg = genrate_bigrams('applying')
# # model_pretrained, predicted_size = my_predict(model  , test_bg)
# # print(model_pretrained)
# # print(predicted_size)


			




		