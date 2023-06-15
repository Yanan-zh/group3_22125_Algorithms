import numpy as np
import random
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import os
import time
import math

import ANN
import PSSM
import SMM

start = time.time()

def mse(y_target_array, y_pred_array):
    return np.sqrt(((y_target_array - y_pred_array)**2).mean())

# with open("./results.txt","a") as f :
#     f.write("MHC\tN_binders\tPSSM_error\tSMM_error\tANN_error\n")

mhc_dir = "./data/"

mhc_list = os.listdir(mhc_dir)
binder_threshold = 1-math.log(500)/math.log(50000)

number_of_binders = []
PSSM_errors = []
SMM_errors = []
ANN_errors = []

for mhc in mhc_list:

    mhc_start = time.time()

    print("Started ", mhc)
    dataset = []

    np.random.seed(11)

    for i in range(5):
        filename = mhc_dir+mhc+"/c00" + str(i)
        dataset.append(np.loadtxt(filename, dtype = str))
        np.random.shuffle(dataset[i])
        dataset[i][:,2] = dataset[i][:,1].astype(float) > binder_threshold
  
    whole_dataset = np.concatenate(dataset, axis = 0)


    prediction_SMM = [None, None, None, None, None]

    for outer_index in range(5) :

        print("\tOuter index: {}/5".format(outer_index+1))

        evaluation_data = dataset[outer_index]

        SMM_matrices = []
        inner_indexes = [i for i in range(5)]
        inner_indexes.remove(outer_index)

        for inner_index in inner_indexes :
        	
        	test_data = dataset[inner_index]

            train_indexes = inner_indexes.copy()
            train_indexes.remove(inner_index)

            train_data = [dataset[i] for i in train_indexes]
            train_data = np.concatenate(train_data, axis = 0)

        	eval_mse = []
        	lamda_optimal = 0 
        	
        	for number in range(lambda_values):
        	
        	
        		print("\t\t\tTraining SMM ...")
        		eval_mse.append(SMM.train(lambda_values[number], train_data, test_data)[1])
        		SMM_matrices.append(SMM.train(train_data, test_data)[0])
        	
			lamda_optimal = lambda_values[np.argmin(eval_mse)]
			
			SMM_matrices_optimal = SMM_matrices[np.argmin(eval_mse)]
			
			evaluation_SMM[inner_index] = np.array(SMM.evaluate(evaluation_data, SMM_matrices_optimal)).reshape(-1,1)
		
		prediction_SMM[outer_index] = np.mean(np.concatenate(evaluation_SMM, axis = 1), axis = 1)
	
	predictions_SMM = np.concatenate(prediction_SMM, axis = 0).reshape(-1,1)
	
    SMM_errors.append(mse(whole_dataset[:,1].astype(float), predictions_SMM[:,0]))
           
       
    with open("./results_SMM.txt","a") as f :
        f.write(mhc+"\t"+str(number_of_binders[-1])+str(SMM_errors[-1]))

    print("\t{} completed in {}s".format(mhc, time.time()-mhc_start))

    print("\n--------------------------\n")
