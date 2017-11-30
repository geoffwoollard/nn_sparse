from __future__ import division
from scipy.sparse.linalg import spsolve
from scipy.sparse import hstack, csr_matrix
from scipy import sparse, diag
import scipy.io
import numpy as np
from math import log
import pickle

class MyClassifier(object):
	import numpy as np
	from math import log
	def __init__(self):
		self.params = ['theta','gamma']
	def fit(self, Xtrain, ytrain,k):
		import numpy as np
		
		def initTheta(X,k):	# random weights _into_ hidden layer, # rows for inputs, # columns for # neuronw
			theta = np.random.randn(X.shape[1],k)
			return theta

		def initGamma(k):	# randomize weights leading _out of_ the hidden layer
			gamma = np.random.randn(k+1)
			return gamma

		def logistic(a):	# sigmoid
			return 1.0/(1+np.exp(-a))

		def linear(y):		# linearizer to [0,1] for Of -> yhat
			y = (y - np.min(y)) / (np.max(y) - np.min(y))
			return y

		def forPass(X,theta,gamma):	# compute the outputs of te neurons given the input and weights
			U = X*theta		# first sum. dot products of instances: shape = (instances, neurons)	
			O = logistic(U)		# output of neurons. element wise logistic
			O1 = np.ones((O.shape[0],O.shape[1]+1))	# add offset column
			O1[:,1:] = O 
			O = O1
			Uf = np.dot(O,gamma)	# final sum
			Of = Uf			
			yhat = linear(Of)	# linearlization to make in [0,1] interval. 
						# by definition will get min yhat = 0 and max yhat = 1.
			return U,O,Uf,yhat


		#########
		######### back propagation
		#########

		def thetaUpdate(theta,y,X,O,gamma,yhat,Uf):
			O = O[:,1:]	# get rid of offset column in O since leads to no neuron and no thetas
			gamma = gamma[1:]
			gammaO = gamma*O*(1-O)
			S = sparse.lil_matrix((X.shape[0],X.shape[0]))
			S.setdiag(y - yhat)
			yXt = (S*X).T	# sparse. shape = (d,miniN)
			step = yXt*gammaO	# (d x k+1)
			step = step/(np.max(Uf) - np.min(Uf))	# extra chain rule term for linearization operation
			theta = theta + step	# step = sum { (y - yhat)*dyhat/dtheta }
			return theta

		def gammaUpdate(gamma,y,O,yhat,Uf,n=1):	# n is step size
			row = yhat*(1-yhat)
			grad = np.multiply(O,row.reshape(-1,1))
			step = np.dot(y - yhat, grad)
			step = step/(np.max(Uf) - np.min(Uf))
			gamma = gamma + n*step
			return gamma

		def fitSlice(ytrain,Xtrain,theta,gamma,numSteps):	# fits theta and gamma to input data. numSteps controls number of times we: 
									# back propagate
									# compute outputs in forward pass
			err = 0
			for tmp in range(numSteps):				# number of time fit data slice ~ steps of gradient
				U,O,Uf,yhat = forPass(Xtrain,theta,gamma)	# update terms for gradient
				err_ = err
				if(1):
					theta_ = theta
					theta = thetaUpdate(theta,ytrain,Xtrain,O,gamma,yhat,Uf)	# update theta
					err = np.sum(np.abs(theta-theta_))
					#print "T", np.min(np.abs(theta-theta_)), np.sum(np.abs(theta-theta_)), err_ - err,theta[0]
				if(1):
					gamma_ = gamma
					gamma = gammaUpdate(gamma,ytrain,O,yhat,Uf,1)	# update gamma
					err = np.sum(np.abs(gamma-gamma_))
					#print "G",gamma[:3],err
				if(0):
					print sum((yhat > 0.5 ) == ytrain)/ytrain.size	# for debugging
			return theta,gamma

		def miniB(y,X,k,numSteps=10, miniN=50000,numEp=5):	# starts with random weights
									# updates weights by fitting to slices of data (miniBatch)
									# miniN is how big slice is
									# numEp is how many times that it passes through all the data
			theta = initTheta(X,k)
			gamma = initGamma(k)
			for EPOCHS in range(numEp):
				new = miniN
				old = 0
				while new <= X.shape[0]:
				#for tmp in range(int(X.shape[0]/miniN)):
					theta, gamma = fitSlice(y[old:new],X[old:new],theta,gamma,numSteps)
					old = new
					new = new + miniN
					U,O,Uf,yhat = forPass(X,theta,gamma)
				#print old,new,sum((yhat > 0.5 ) == y)/y.size
			return theta,gamma

		NEURONS = k
		miniN = 50000
		numSteps = 10
		numEp = 3
		self.theta,self.gamma = miniB(ytrain,Xtrain,NEURONS,numSteps, miniN,numEp)	
			# values found by cross validation, expect ~ 75 % accurac on train and test sets
			# training takes ~ 10 minutes

	def predict(self, X):	# see fit for comments of these functions
		def logistic(a):
			return 1.0/(1+np.exp(-a))

		def linear(y):
			y = (y - np.min(y)) / (np.max(y) - np.min(y))
			return y
		def forPass(X,theta,gamma):
			U = X*theta	# dot products of instances: shape = (instances, neurons)	
			O = logistic(U)		# element wise logistic
			O1 = np.ones((O.shape[0],O.shape[1]+1))
			O1[:,1:] = O
			O = O1
			Uf = np.dot(O,gamma)
			Of = Uf
			yhat = linear(Of)
			return U,O,Uf,yhat
		U,O,Uf,yhat = forPass(X,self.theta,self.gamma)
		yhat = yhat > 0.5
		yhat = yhat.astype(float)		# convert y to 0s and 1s
		return yhat
		
	def save_params(self, fname):
		params = dict([(p, getattr(self, p)) for p in self.params])
		np.savez(fname, **params)
	
	def load_params(self, fname):
		params = np.load(fname)
		for name in self.params:
			setattr(self, name, params[name])
