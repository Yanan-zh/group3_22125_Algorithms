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
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import numpy as np
from torch.autograd import Variable
import time

# %%
from pytorchtools import EarlyStopping

import os
import time
import math



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
    

def mse(y_true, y_pred):
    return torch.mean((y_true - y_pred) ** 2)


def pcc(y_true, y_pred):
    mean_true = torch.mean(y_true)
    mean_pred = torch.mean(y_pred)
    centered_true = y_true - mean_true
    centered_pred = y_pred - mean_pred
    pcc_num = torch.sum(centered_true * centered_pred)
    pcc_den = torch.sqrt(torch.sum(centered_true ** 2) * torch.sum(centered_pred ** 2))
    return pcc_num / pcc_den

def Tensorprocess_valid(valid_raw):
    x_valid_, y_valid_ = encode_peptides(valid_raw)
    x_valid_ = x_valid_.reshape(x_valid_.shape[0], -1)
    x_valid = Variable(torch.from_numpy(x_valid_.astype('float32')))
    y_valid = Variable(torch.from_numpy(y_valid_.astype('float32'))).view(-1, 1)
    return x_valid, y_valid


def Tensorprocess_test(test_raw):
    x_test, y_test = encode_peptides(test_raw)
    x_test = x_test.reshape(x_test.shape[0], -1)
    x_test = Variable(torch.from_numpy(x_test.astype('float32')))
    y_test = Variable(torch.from_numpy(y_test.astype('float32'))).view(-1, 1)
    return x_test, y_test

def evaluate_model(x_eval, y_eval, net):
    net.eval()
    output = net(x_eval)
    return output

def random_split_outer(data, num_splits=5):
    splits = []

    for _ in range(num_splits):
        train, test = train_test_split(data, test_size=0.2, random_state=42)
        splits.append((train, test))

    return splits

def random_split_inner(data, num_splits=4):
    splits = []

    for _ in range(num_splits):
        train, test = train_test_split(data, test_size=0.2, random_state=42)
        splits.append((train, test))

    return splits

# No mini-batch loading
# mini-batch loading
def train_model(x_train, y_train, net, optimizer, criterion):
    net.train()
    optimizer.zero_grad()
    output = net(x_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    return net


# with open("./results.txt","a") as f :
#     f.write("MHC\tN_binders\tANN_error\tANN_error\tANN_error\n")

mhc_dir = "/Users/einar/Documents/algo/code/Project/data/"
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



# Define your parameters
N_HIDDEN_NEURONS_param = [2, 4, 6, 16, 64]
learning_rate_param = [0.01, 0.05, 0.1]

# Define the directory to save the best model and results
results_dir = "./results/"

# Create the directory if it doesn't exist
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Loop over each MHC
for mhc in mhc_list:
    mhc_start = time.time()
    print("Started ", mhc)

    dataset = pd.DataFrame()  # Initialize an empty DataFrame

    for i in range(5):
        filename = mhc_dir + mhc + "/c00" + str(i)
        print(filename)
        df = pd.read_csv(filename, sep='\s+', usecols=[0, 1], names=['peptide', 'target'])
        df = df.sort_values(by='target', ascending=False).reset_index(drop=True)
        dataset = pd.concat([dataset, df], axis=0)

    dataset = dataset.reset_index(drop=True)  # Reset the index of the concatenated DataFrame
    split = random_split_outer(dataset)
    print(dataset)

    best_mse = float('inf')
    best_pcc = float('-inf')
    best_model_path = ""

    for outer_index in range(5):
        print("\tOuter index: {}/5".format(outer_index + 1))

        evaluation_data = split[outer_index][1]
        evaluation_data = evaluation_data.sort_values(by='target', ascending=False).reset_index(drop=True)
        x_test, y_test = Tensorprocess_test(evaluation_data)

        inner_split = random_split_inner(split[outer_index][0])

        for inner_index in range(4):
            test_data = inner_split[inner_index][1]
            test_data = test_data.sort_values(by='target', ascending=False).reset_index(drop=True)
            x_valid, y_valid = Tensorprocess_valid(test_data)

            train_data = inner_split[inner_index][0]
            train_data = train_data.sort_values(by='target', ascending=False).reset_index(drop=True)
            x_train_, y_train_ = encode_peptides(train_data)
            x_train_ = x_train_.reshape(x_train_.shape[0], -1)
            batch_size = x_train_.shape[0]
            n_features = x_train_.shape[1]
            x_train = Variable(torch.from_numpy(x_train_.astype('float32')))
            y_train = Variable(torch.from_numpy(y_train_.astype('float32'))).view(-1, 1)

            for number1 in range(len(N_HIDDEN_NEURONS_param)):
                for number2 in range(len(learning_rate_param)):
                    print("\t\t\tTraining ANN ...")
                    N_HIDDEN_NEURONS = N_HIDDEN_NEURONS_param[number1]
                    learning_rate = learning_rate_param[number2]
                    net = Net(n_features, N_HIDDEN_NEURONS)
                    optimizer = optim.SGD(net.parameters(), lr=learning_rate)
                    criterion = nn.MSELoss()

                    for epoch in range(100):
                        net = train_model(x_train, y_train, net, optimizer, criterion)

                    # Evaluate the model
                    with torch.no_grad():
                        y_pred = evaluate_model(x_valid, y_valid, net)
                        current_mse = mse(y_valid, y_pred)
                        current_pcc = pcc(y_valid, y_pred)

                        # Check if the current model is the best
                        if current_mse < best_mse:
                            best_mse = current_mse
                            best_pcc = current_pcc
                            best_model_path = os.path.join(results_dir, mhc + "_best_model.pth")
                            torch.save(net.state_dict(), best_model_path)

    # Write the results to a file
    results_file_path = os.path.join(results_dir, mhc + "_results.txt")
    with open(results_file_path, "w") as f:
        f.write("MHC: {}\n".format(mhc))
        f.write("Best Model: {}\n".format(best_model_path))
        f.write("Best MSE: {}\n".format(best_mse))
        f.write("Best PCC: {}\n".format(best_pcc))

    print("\t{} completed in {}s".format(mhc, time.time() - mhc_start))
    print("\n--------------------------\n")
# %%
