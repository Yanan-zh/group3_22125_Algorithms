# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, matthews_corrcoef
from scipy.stats import pearsonr
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# %%
from pytorchtools import EarlyStopping

import os
import time
import math

#import ANN
import ANN_new


start = time.time()

def error(y, y_pred):
    return 0.5*(y_pred - y)**2

def load_blosum(filename):
    """
    Read in BLOSUM values into matrix.
    """
    aa = ['A', 'R', 'N' ,'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'X']
    df = pd.read_csv(filename, sep='\s+', comment='#', index_col=0)
    return df.loc[aa, aa]

def load_peptide_target(filename):
    """
    Read amino acid sequence of peptides and
    corresponding log transformed IC50 binding values from text file.
    """
    df = pd.read_csv(filename, sep='\s+', usecols=[0,1], names=['peptide','target'])
    return df.sort_values(by='target', ascending=False).reset_index(drop=True)

def encode_peptides(Xin):
    """
    Encode AA seq of peptides using BLOSUM50.
    Returns a tensor of encoded peptides of shape (batch_size, MAX_PEP_SEQ_LEN, n_features)
    """
    blosum = load_blosum(blosum_file)
    
    batch_size = len(Xin)
    n_features = len(blosum)
    
    Xout = np.zeros((batch_size, MAX_PEP_SEQ_LEN, n_features), dtype=np.int8)
    
    for peptide_index, row in Xin.iterrows():
        for aa_index in range(len(row.peptide)):
            Xout[peptide_index, aa_index] = blosum[ row.peptide[aa_index] ].values
            
    return Xout, Xin.target.values

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def invoke(early_stopping, loss, model, implement=False):
    if implement == False:
        return False
    else:
        early_stopping(loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            return True
        
class Net(nn.Module):

    def __init__(self, n_features, n_l1):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_features, n_l1)
        self.fc2 = nn.Linear(n_l1, 1)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
def init_weights(m):
    """
    https://pytorch.org/docs/master/nn.init.html
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.1) # nn.init.constant_(m.bias, 0)

def pearson(pred, targ):
    p = pred - pred.mean()
    t = targ - targ.mean()
    s = p.mul(t).sum()
    s1 = p.mul(p).sum().sqrt()
    s2 = t.mul(t).sum().sqrt()
    return s / ( s1 * s2)

def Tensorprocess_test(valid_data):
    valid_raw = load_peptide_target(valid_data)
    x_valid_, y_valid_ = encode_peptides(valid_raw)
    x_valid_ = x_valid_.reshape(x_valid_.shape[0], -1)
    x_valid = Variable(torch.from_numpy(x_valid_.astype('float32')))
    y_valid = Variable(torch.from_numpy(y_valid_.astype('float32'))).view(-1, 1)
    return x_valid, y_valid

def Tensorprocess_train(train_data,):
    train_raw = load_peptide_target(train_data)
    x_train_, y_train_ = encode_peptides(train_raw)
    x_train_ = x_train_.reshape(x_train_.shape[0], -1)
    batch_size = x_train_.shape[0]
    n_features = x_train_.shape[1]
    x_train = Variable(torch.from_numpy(x_train_.astype('float32')))
    y_train = Variable(torch.from_numpy(y_train_.astype('float32'))).view(-1, 1)
    return x_train, y_train

# No mini-batch loading
# mini-batch loading
def train():
    train_loss, valid_loss = [], []

    early_stopping = EarlyStopping(patience=PATIENCE)

    for epoch in range(EPOCHS):
        net.train()
        pred = net(x_train)
        loss = criterion(pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.data)

        if epoch % (EPOCHS//10) == 0:
            print('Train Epoch: {}\tLoss: {:.6f}'.format(epoch, loss.data))

        net.eval()
        pred = net(x_valid)
        loss = criterion(pred, y_valid)  
        valid_loss.append(loss.data)
        mse = torch.mean(error(pred.detach(), y_valid.detach()))
        pcc = pearson(pred.detach(), y_valid.detach())

        if invoke(early_stopping, valid_loss[-1], net, implement=True):
            net.load_state_dict(torch.load('checkpoint.pt'))
            break
            
    return net, train_loss, valid_loss, mse, pcc


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
blosum_file = "/Users/einar/Documents/algo/data/Matrices/BLOSUM62"

mhc_list = os.listdir(mhc_dir)
binder_threshold = 0.426
MAX_PEP_SEQ_LEN = 9
EPOCHS = 3000
MINI_BATCH_SIZE = 512
PATIENCE = EPOCHS // 10

number_of_binders = []
ANN_errors = []
ANN_errors = []
ANN_errors = []



N_HIDDEN_NEURONS = np.array([2, 4, 6, 16, 64])
learning_rate = np.array([0.1, 0.05, 0.01])


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
        evaluation_data = Tensorprocess_test(evaluation_data)

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
            train_data = Tensorprocess_test(train_data)

            
            nh_optimal = 0
            learn_optimal = 0
            test_mse_list = []
            for number1 in range(len(N_HIDDEN_NEURONS)):
            
            	for number2 in range(len(learning_rate)):
            	
                	print("\t\t\tTraining ANN ...")
                    N_HIDDEN_NEURONS[number1]
                    learning_rate[number2]
                    net = Net(n_features, N_HIDDEN_NEURONS)
                    optimizer = optim.SGD(net.parameters(), lr=learning_rate)
                    criterion = nn.MSELoss()
                	weight_matrix, train_loss, valid_loss, mse, pcc = train()
                	# print(test_mse)
                	test_mse_list.append(mse)
                	ANN_matrices.append(weight_matrix)
            
            nh_optimal = N_HIDDEN_NEURONS[np.argmin(test_mse_list)// 3]
            learn_optimal = learning_rate[np.argmin(test_mse_list)% 3]
            
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