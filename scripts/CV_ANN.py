
import numpy as np
import random
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import os
import time
import math

#import ANN
import ANN_new

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
#     f.write("MHC\tN_binders\tANN_error\tANN_error\tANN_error\n")

mhc_dir = "../data/"

mhc_list = os.listdir(mhc_dir)
binder_threshold = 0.426

number_of_binders = []
ANN_errors = []
ANN_errors = []
ANN_errors = []


N_HIDDEN_NEURONS = np.array([2, 4, 6, 16, 64])
learning_rates = np.array([0.1, 0.05, 0.01])


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


    prediction_ANN = [None, None, None, None, None]


    for outer_index in range(5) :

        print("\tOuter index: {}/5".format(outer_index+1))

        evaluation_data = dataset[outer_index]

        ANN_matrices = []
        inner_indexes = [i for i in range(5)]
        inner_indexes.remove(outer_index)
        
        evaluation_ANN = []
        


        for inner_index in inner_indexes :
            
            test_data = dataset[inner_index]

            train_indexes = inner_indexes.copy()
            train_indexes.remove(inner_index)

            train_data = [dataset[i] for i in train_indexes]
            train_data = np.concatenate(train_data, axis = 0)

            
            nh_optimal = 0
            learn_optimal = 0
            test_mse_list = []
            for number1 in range(len(N_HIDDEN_NEURONS)):
            
            	for number2 in range(len(learning_rates)):
            	
                	print("\t\t\tTraining ANN ...")
                
                	test_mse, weight_matrix = ANN.train(train_data, test_data, N_HIDDEN_NEURONS[number1],learning_rate[number2])
                	# print(test_mse)
                	test_mse_list.append(test_mse)
                	ANN_matrices.append(weight_matrix)
            
            nh_optimal = N_HIDDEN_NEURONS[np.argmin(test_mse_list)// 3]
            learn_optimal = learning_rates[np.argmin(test_mse_list)% 3]
            
            ANN_matrices_optimal = ANN_matrices[np.argmin(test_mse_list)]

            evaluation_ANN.append(np.array(evaluate(evaluation_data, ANN_matrices_optimal)).reshape(-1,1))

        prediction_ANN[outer_index] = np.mean(np.concatenate(evaluation_ANN, axis = 1), axis = 1)
        
    predictions_ANN = np.concatenate(prediction_ANN, axis = 0).reshape(-1,1)
    print(min(predictions_ANN))
    
    ANN_errors.append(mse(whole_dataset[:,1].astype(float), predictions_ANN[:,0]))
           
       
#    with open("./results_ANN.txt","a") as f :
 #       f.write(mhc+"\t"+str(number_of_binders[-1])+str(ANN_errors[-1]))

    print("\t{} completed in {}s".format(mhc, time.time()-mhc_start))

    print("\n--------------------------\n")
    
print(ANN_errors)