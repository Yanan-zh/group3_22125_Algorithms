#!/usr/bin/env python

import numpy as np
from scipy.stats import pearsonr


#######################
alphabet = np.loadtxt('../Matrices/alphabet',dtype=str)
_blosum62 = np.loadtxt('../Matrices/blosum62.freq_rownorm', dtype=float).reshape((20,20)).T

blosum62 = {}
for i, letter_1 in enumerate(alphabet):
	blosum62[letter_1] = {}
	for j, letter_2 in enumerate(alphabet):
		blosum62[letter_1][letter_2] = _blosum62[i,j]
#######################


sparse_file = "../Matrices/sparse"
_sparse = np.loadtxt(sparse_file, dtype=float)
sparse = {}

for i, letter_1 in enumerate(alphabet):
    
    sparse[letter_1] = {}

    for j, letter_2 in enumerate(alphabet):
        
        sparse[letter_1][letter_2] = _sparse[i, j]

###################
def encode(peptides, encoding_scheme, alphabet):

	encoded_peptides = []

	for peptide in peptides:

		encoded_peptide = []

		for peptide_letter in peptide:

			for alphabet_letter in alphabet:
				encoded_peptide.append(encoding_scheme[peptide_letter][alphabet_letter])

		encoded_peptides.append(encoded_peptide)
        
	return np.array(encoded_peptides)

def cummulative_error(peptides, y, lamb, weights):

    error = 0
    
    for i in range(0, len(peptides)):
        
        # get peptide
        peptide = peptides[i]

        # get target prediction value
        y_target = y[i]
        
        # get prediction
        y_pred = np.dot(peptide, weights)
            
        # calculate error
        error += 1.0/2 * (y_pred - y_target)**2
        
    gerror = error + lamb*np.dot(weights, weights)
    error /= len(peptides)
        
    return gerror, error


def predict(peptides, weights):
	
	pred = []

	for i in range(0, len(peptides)):

		peptide = peptides[i]

		y_pred = np.dot(peptide, weights)

		pred.append(y_pred)

	return pred
def cal_mse(vec1, vec2):
    
    mse = 0
    
    for i in range(0, len(vec1)):
        mse += (vec1[i] - vec2[i])**2
        
    mse /= len(vec1)
    
    return( mse)

def gradient_descent(y_pred, y_target, peptide, weights, lamb_N, epsilon):

	do = y_pred - y_target

	for i in range(0,len(weights)):
		de_dw_i = do*peptide[i] + (2*lamb_N)*weights[i]

		#print(de_dw_i)
		weights[i] -= epsilon * de_dw_i


def train(training_data, test_data, lamb):
#	training_data = np.loadtxt(training_file,dtype=str)
#	test_data = np.loadtxt(test_file, dtype=str)

	peptides = training_data[:,0]
	peptides = encode(peptides, sparse, alphabet)
	N = len(peptides)


	#target value
	y = np.array(training_data[:,1], dtype = float)

	# test peptide
	test_peptides = test_data[:,0]
	test_peptides = encode(test_peptides, sparse, alphabet)

	# test targets
	test_targets = np.array(test_data[:,1],dtype=float)

	#weights
	input_dim = len(peptides[0])
	output_dim = 1
	w_bound = 0.1
	weights = np.random.uniform(-w_bound, w_bound, size=input_dim)

	#training epochs
	epochs = 100

	#regularization lambda per target value
	lamb_N = lamb/N

	#learning rate
	epsilon = 0.01

	for e in range(0,epochs):

		for i in range(0,N):

			ix = np.random.randint(0,N)

			peptide = peptides[ix]
			#print(peptide)

			y_target = y[ix]

			y_pred = np.dot(peptide,weights)

			#print(weights)


			gradient_descent(y_pred, y_target, peptide, weights, lamb_N, epsilon)

			#print(ix, peptide, y_target, y_pred,weights)
			#exit()

		gerr, mse = cummulative_error(peptides, y, lamb, weights)


		#train_pred = predict( peptides, weights )
		#train_mse = cal_mse( y, train_pred )
		#train_pcc = pearsonr( y, train_pred )

		# predict on test data
		test_pred = predict(test_peptides, weights)
		test_mse = cal_mse(test_targets,test_pred)
		#print(np.min(test_pred), np.mean(test_pred), np.max(test_pred))
		test_pcc = pearsonr(test_targets, test_pred)
		#print ("Epoch: ", e, "Gerr:", gerr, train_pcc[0], train_mse, test_pcc[0], test_mse)
	return lamb, test_mse, weights, test_pred

def evaluate(evaluation_data, weight):
	peptides = evaluation_data[:,0]
	peptides = encode(peptides, sparse, alphabet)
	prediction = predict(peptides,weight)

	return prediction


#data_dir = "/home/jich/Algo/data/"
#training_file = data_dir + "SMM/A0201_training"
#evaluation_file = data_dir + "SMM/A0201_evaluation"

#np.random.seed(1)

#training_data = np.loadtxt(training_file, dtype=str)
#test_data = np.loadtxt(evaluation_file, dtype=str)
#train(training_data,test_data,0.01)

