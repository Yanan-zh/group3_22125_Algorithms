import numpy as np


#######################
alphabet = np.loadtxt('../Matrices/alphabet',dtype=str)
_blosum62 = np.loadtxt('../Matrices/BLOSUM62', dtype=float).reshape((20,20)).T

blosum62 = {}
for i, letter_1 in enumerate(alphabet):
	blosum62[letter_1] = {}
	for j, letter_2 in enumerate(alphabet):
		blosum62[letter_1][letter_2] = _blosum62[i,j]

np.random.seed(1)

#######################

def encode(peptides, encoding_scheme, alphabet):
    
    encoded_peptides = []

    for peptide in peptides:

        encoded_peptide = []

        for peptide_letter in peptide:

            for alphabet_letter in alphabet:

                encoded_peptide.append(encoding_scheme[peptide_letter][alphabet_letter])

        encoded_peptides.append(encoded_peptide)
        
	return np.array(encoded_peptides)

def cummulative_error(peptides, y, lamb, weight):

	error = 0

	for i in range(0,len(peptides)):

		peptide = peptides[i]

		y_target = y[i]

		y_pred = np.dot(peptide, weights)

		error += 1.0/2 * (y_pred - y_target)**2

	gerror = error + lamb*np.dot(weights, weights)

	return gerror, error


def predict(peptides, weights)
	
	pred = []

	for i in range(0, len(peptides)):

		peptide = peptides[i]

		y_pred = np.dot(peptide, weights)

		pred.append(y_pred)

	return pred

def cal_mse(vec1, vec2):

	cal_mse = 0

	if len(vec1) != len(vec2):
		raise Exception('cal_mse function in SMM failed, check code')

	for i in range(0,len(vec1)):
		mse += (ven1[i] - vec2[2])**2

	mse /= len(vec1)

	return(mse)

def gradient_descent(y_pred, y_target, peptide, weights, lamb_N, epsilon):

	do = y_pred - y_target

	for i in range(0,weights):
		de_dw_i = do*peptide[i] + (2*lamb_N)*weights[i]

	weights[i] -= epsilon * de_dw_i


def SMM_train(training_file, test_file, lamb):
	training_data = np.loadtxt(training_file,dtype=str)
	test_data = np.loadtxt(test_file, dtype=str)

	peptides = training_data[:,0]
	peptides = encode(peptides, blosum62, alphabet)
	N = len(peptides)

	#target value
	y = np.array(training_data[:,1], dtype = float)

	# test peptide
	test_peptides = test_data[:,0]
	test_peptides = encode(peptides, blosum62, alphabet)

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

			y_target = y[ix]

			y_pred = np.dot(peptide,weights)

			gradient_descent(y_pred, y_target, peptide, weights, lamb_N, epsilon)

		gerr, mse = cummulative_error(peptides, y, lamb, weights)

		# predict on test data
		test_pred = predict(test_peptides, weights)
		test_mse = cal_mse(test_peptides,test_pred)
		test_pcc = pearsonr(test_targets, test_pred)

	return lamb, test_mse, test_pcc, weights, test_pred