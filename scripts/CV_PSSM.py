import numpy as np
import random
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import os
import time
import math

#import ANN
import PSSM_new

start = time.time()

def mse(y_target_array, y_pred_array):
    return np.sqrt(((y_target_array - y_pred_array)**2).mean())


def evaluate(evaluation, w_matrix):
  
  def score_peptide(peptide, matrix):
    acum = 0
    for i in range(0, len(peptide)):
        acum += matrix[i][peptide[i]]
    return acum

  # Read evaluation data
  # evaluation = np.loadtxt(evaluation_file, dtype=str).reshape(-1,2)
  evaluation_peptides = evaluation[:, 0]
  evaluation_targets = evaluation[:, 1].astype(float)

  evaluation_peptides, evaluation_targets

  peptide_length = len(evaluation_peptides[0])

  evaluation_predictions = []
  for i in range(len(evaluation_peptides)):
    score = score_peptide(evaluation_peptides[i], w_matrix)
    evaluation_predictions.append(score)
    #print (evaluation_peptides[i], score, evaluation_targets[i])
  return(evaluation_predictions)





# with open("./results.txt","a") as f :
#     f.write("MHC\tN_binders\tPSSM_error\tPSSM_error\tANN_error\n")

mhc_dir = "../data/"

mhc_list = os.listdir(mhc_dir)
binder_threshold = 0.426

number_of_binders = []
PSSM_errors = []
PSSM_errors = []
ANN_errors = []

beta_values = np.array([0, 50, 100, 150, 200, 250, 300])


for mhc in mhc_list:

    mhc_start = time.time()

    print("Started ", mhc)
    dataset = []

    np.random.seed(11)

    for i in range(5):
        filename = mhc_dir+mhc+"/c00" + str(i)
        print(filename)
        dataset.append(np.loadtxt(filename, dtype = str))
        np.random.shuffle(dataset[i])
        dataset[i][:,2] = dataset[i][:,1].astype(float) > binder_threshold
  
    whole_dataset = np.concatenate(dataset, axis = 0)


    prediction_PSSM = [None, None, None, None, None]
    evaluation_PSSM = [None, None, None, None]

    for outer_index in range(5) :

        print("\tOuter index: {}/5".format(outer_index+1))

        evaluation_data = dataset[outer_index]

        PSSM_matrices = []
        inner_indexes = [i for i in range(5)]
        inner_indexes.remove(outer_index)

        for inner_index in inner_indexes :
            
            test_data = dataset[inner_index]

            train_indexes = inner_indexes.copy()
            train_indexes.remove(inner_index)

            train_data = [dataset[i] for i in train_indexes]
            train_data = np.concatenate(train_data, axis = 0)

            test_mse_list = []
            beta_optimal = 0 
            eval_mse = []
            for number in range(len(beta_values)):
            
                print("\t\t\tTraining PSSM ...")
                weight = PSSM_new.PSSM_train(train_data, beta_values[number])
                PSSM_matrices.append(weight)
                 
                pred = evaluate(test_data, weight)
                targets = np.array(test_data[:, 1], dtype=float)
                error = mse(targets,pred)
                eval_mse.append(error)
            
            beta_optimal = beta_values[np.argmin(eval_mse)]
            
            PSSM_matrices_optimal = PSSM_matrices[np.argmin(eval_mse)]


            evaluation_PSSM[inner_index-1] = np.array(evaluate(evaluation_data, PSSM_matrices_optimal)).reshape(-1,1)

        prediction_PSSM[outer_index] = np.mean(np.concatenate(evaluation_PSSM, axis = 1), axis = 1)
        print(evaluation_PSSM)
    predictions_PSSM = np.concatenate(prediction_PSSM, axis = 0).reshape(-1,1)
    
    PSSM_errors.append(mse(whole_dataset[:,1].astype(float), predictions_PSSM[:,0]))
           
       
    with open("./results_PSSM.txt","a") as f :
        f.write(mhc+"\t"+str(number_of_binders[-1])+str(PSSM_errors[-1]))

    print("\t{} completed in {}s".format(mhc, time.time()-mhc_start))

    print("\n--------------------------\n")
