from __future__ import division
from scipy.sparse.linalg import spsolve
from scipy.sparse import hstack, csr_matrix
from scipy import sparse, diag
import scipy.io
import numpy as np
from math import log
import pickle
from classify import nn_sparse_classifier

def load_data_from_pkl(file_name='data/data.pkl'):
	pkl_file = open(file_name,'rb')
	data = pickle.load(pkl_file)
	return(data)

def randSp(Msp,random_seed=0):
	d = np.arange(np.shape(Msp)[0])
	np.random.seed(random_seed)
	np.random.shuffle(d)
	Mrand =  Msp[d, :]
	return Mrand

# todo: return dataframe of stats, and best params, runtime
# todo: save params with hyper params in filename
def main():

	data = load_data_from_pkl(file_name='data/data.pkl')
	dataR = randSp(data)
	
	X = dataR[:,1:]					# (1000000, 10063)
	y = dataR[:,0]	
	y = np.array(y.todense()).flatten()

	# offset 
		#x1 = csr_matrix(np.ones(X.shape[0]).reshape(-1,1))
		#X = csr_matrix(hstack( [epsilon*x1,X] ))
	
	trN=900000
	Xtrain=X[:trN,:]
	ytrain=y[:trN]
	Xtest = X[trN:,:]
	ytest = y[trN:]
	
	
	classifier = nn_sparse_classifier()
	for k in range(20,40,5):
		print k,
		if(1):
			classifier.fit(Xtrain,ytrain,k)
			classifier.save_params('data/paramsNew.npz')
		if(1):
			classifier.load_params('data/paramsNew.npz')
			yhat = classifier.predict(Xtrain)
			print sum(yhat == ytrain)/yhat.size,
			yhat = classifier.predict(Xtest)
			print sum(yhat == ytest)/yhat.size
			
			
if __name__ == "__main__":
	main()
	

# cross validation
#3 0.758406666667 0.74648
#4 0.762307777778 0.74781
#5 0.761155555556 0.74491
#6 0.760576666667 0.74607
#7 0.76579 0.75044
#8 0.768302222222 0.7549
#9 0.766022222222 0.75304
#10 0.770022222222 0.75609
#11 0.76994 0.75574
#12 0.76858 0.75391
#13 0.771351111111 0.75736
#14 0.771258888889 0.751
#15 0.770935555556 0.75779
#16 0.769733333333 0.75696
#17 0.770744444444 0.75783
#18 0.771171111111 0.75914
#19 0.770772222222 0.75831
	

