#!/usr/bin/env python

import numpy as np
import random
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import os
import time
import math

import sys

#import ANN
#import PSSM
import SMM_Gradient_Descent as SMM

start = time.time()


alphabet = np.loadtxt('../Matrices/alphabet',dtype=str)
_blosum62 = np.loadtxt('../Matrices/blosum62.freq_rownorm', dtype=float).reshape((20,20)).T

blosum62 = {}
for i, letter_1 in enumerate(alphabet):
    blosum62[letter_1] = {}
    for j, letter_2 in enumerate(alphabet):
        blosum62[letter_1][letter_2] = _blosum62[i,j]

def mse(y_target_array, y_pred_array):

    mse = 0

    for i in range(0,len(y_target_array)):
        mse += 1/2*(y_target_array[i] - y_pred_array[i])**2

    mse /= len(y_target_array)

    return mse

with open("./results.txt","a") as f :
     f.write("MHC\tN_binders\tPSSM_error\tSMM_error\tANN_error\n")

mhc_dir = "../data/"
mhc_list = os.listdir(mhc_dir)
binder_threshold = 1-math.log(500)/math.log(50000)

#test = []
number_of_binders = []
pred_binder_number = []
PSSM_errors = []
SMM_errors = []
ANN_errors = []


lambda_values = np.array([0,10**(-3),10**(-2),5*10**(-2),10**(-1),5*10**(-1),5,10,5*10,10**2])
for lambda_loop, lambda_value in enumerate(lambda_values):

    for mhc_loop, mhc in enumerate(mhc_list):

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
        number_of_binders.append(np.count_nonzero(whole_dataset[:][:,2] == 'True'))


        prediction_SMM = [None, None, None, None, None]
    

        for outer_index in range(5) :

            #print("\tOuter index: {}/5".format(outer_index+1))

            evaluation_data = dataset[outer_index]

            evaluation_SMM = []
            inner_indexes = [i for i in range(5)]
            inner_indexes.remove(outer_index)

            for inner_index in inner_indexes :

                test_data = dataset[inner_index]

                train_indexes = inner_indexes.copy()
                train_indexes.remove(inner_index)

                train_data = [dataset[i] for i in train_indexes]
                train_data = np.concatenate(train_data, axis = 0)


                print("\t\t\t Lambda {}, file {}/35, outer index: {}/5, inner index: {}, Training SMM ...".format(lambda_loop+1, mhc_loop+1, outer_index+1, inner_index))

                lamb, test_mse, weights, test_pred = SMM.train(train_data, test_data, lambda_value)
            
            #test.append('lambdavalue: {}, mse: {}'.format(lamda_optimal, min(test_mse_list)))
                evaluation_SMM.append(np.array(SMM.evaluate(evaluation_data, weights)).reshape(-1,1))
            #print(evaluation_SMM)

            prediction_SMM[outer_index] = np.mean(np.concatenate(evaluation_SMM, axis = 1), axis = 1)
        #prediction_SMM[outer_index] = np.mean(np.row_stack(evaluation_SMM), axis = 0)
#        print(prediction_SMM)

    
        predictions_SMM = np.concatenate(prediction_SMM, axis = 0).reshape(-1,1)
    
        SMM_errors.append(mse(whole_dataset[:,1].astype(float), predictions_SMM[:,0]))

        pred_binder_number.append((predictions_SMM > binder_threshold).sum())

        eval_pcc = pearsonr(whole_dataset[:,1].astype(float), predictions_SMM[:,0].astype(float))
        print(eval_pcc)

        with open("./results_SMM_lambda.txt","a") as f :
            f.write(mhc+"\t"+ str(len(whole_dataset)) + '\t' + str(number_of_binders[-1]) + '\t' + str(pred_binder_number[-1]) + '\t' + str(lambda_value) + '\t' + str(SMM_errors[-1]) + '\t' + str(eval_pcc[0]) + '\t' + str(eval_pcc[1]) + '\n')


        print("\t{} completed in {}s".format(mhc, time.time()-mhc_start))

        print("\n--------------------------\n")

