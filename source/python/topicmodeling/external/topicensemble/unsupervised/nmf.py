import numpy as np
from sklearn import decomposition
import logging as log

# --------------------------------------------------------------

class SklNMF:
	"""
	Wrapper class backed by the scikit-learn package NMF implementation.
	"""
	def __init__( self, max_iters = 100, init_strategy = "random" ):
		self.max_iters = 100
		self.init_strategy = init_strategy
		self.W = None
		self.H = None

	def apply( self, X, k = 2, init_W = None, init_H = None ):
		"""
		Apply NMF to the specified document-term matrix X.
		"""
		self.W = None
		self.H = None
		random_seed = np.random.randint( 1, 100000 )
		if not (init_W is None or init_H is None):
			model = decomposition.NMF( init="custom", n_components=k, max_iter=self.max_iters, random_state = random_seed )
			self.W = model.fit_transform( X, W=init_W, H=init_H )
		else:
			model = decomposition.NMF( init=self.init_strategy, n_components=k, max_iter=self.max_iters, random_state = random_seed )
			self.W = model.fit_transform( X )
		self.H = model.components_			
		
	def rank_terms( self, topic_index, top = -1 ):
		"""
		Return the top ranked terms for the specified topic, generated during the last NMF run.
		"""
		if self.H is None:
			raise ValueError("No results for previous run available")
		# NB: reverse
		top_indices = np.argsort( self.H[topic_index,:] )[::-1]
		# truncate if necessary
		if top < 1 or top > len(top_indices):
			return top_indices
		return top_indices[0:top]

	def generate_partition( self ):
		if self.W is None:
			raise ValueError("No results for previous run available")
		return np.argmax( self.W, axis = 1 ).flatten().tolist()		



