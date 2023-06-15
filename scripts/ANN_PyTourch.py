# %%
import torch
import math
from torch.autograd import Variable
import torch.nn as nn
#import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# %%
from pytorchtools import EarlyStopping

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, matthews_corrcoef

# %%
SEED=1
np.random.seed(SEED)
torch.manual_seed(SEED)

# %%
def load_blosum(filename):
    """
    Read in BLOSUM values into matrix.
    """
    aa = ['A', 'R', 'N' ,'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'X']
    df = pd.read_csv(filename, sep='\s+', comment='#', index_col=0)
    return df.loc[aa, aa]

# %%
def load_peptide_target(filename):
    """
    Read amino acid sequence of peptides and
    corresponding log transformed IC50 binding values from text file.
    """
    df = pd.read_csv(filename, sep='\s+', usecols=[0,1], names=['peptide','target'])
    return df.sort_values(by='target', ascending=False).reset_index(drop=True)

# %%
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

# %%
def invoke(early_stopping, loss, model, implement=False):
    if implement == False:
        return False
    else:
        early_stopping(loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            return True

# %%
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# %% [markdown]
# ## Arguments

# %%
MAX_PEP_SEQ_LEN = 9
BINDER_THRESHOLD = 0.426

# %% [markdown]
# # Main

# %% [markdown]
# ## Load

# %%
blosum_file = "/Users/einar/Documents/algo/data/Matrices/BLOSUM62"
train_data = "/Users/einar/Documents/algo/data/SMM/A0201/f000"
valid_data = "/Users/einar/Documents/algo/data/SMM/A0201/c000"

# %%
train_raw = load_peptide_target(train_data)
valid_raw = load_peptide_target(valid_data)

# %% [markdown]
# ### Visualize Data

# %%
def plot_peptide_distribution(raw_data, raw_set):
    raw_data['peptide_length'] = raw_data.peptide.str.len()
    raw_data['target_binary'] = (raw_data.target >= BINDER_THRESHOLD).astype(int)

    # Position of bars on x-axis
    ind = np.arange(train_raw.peptide.str.len().nunique())
    neg = raw_data[raw_data.target_binary == 0].peptide_length.value_counts().sort_index()
    pos = raw_data[raw_data.target_binary == 1].peptide_length.value_counts().sort_index()

    # Plotting
    plt.figure()
    width = 0.3  

    plt.bar(ind, neg, width, label='Non-binders')
    plt.bar(ind + width, pos, width, label='Binders')

    plt.xlabel('Peptide lengths')
    plt.ylabel('Count of peptides')
    plt.title('Distribution of peptide lengths in %s data' %raw_set)
    plt.xticks(ind + width / 2, ['%dmer' %i for i in neg.index])
    plt.legend(loc='best')
    plt.show()

# %% [markdown]
# ### Encode data

# %%
x_train_, y_train_ = encode_peptides(train_raw)
x_valid_, y_valid_ = encode_peptides(valid_raw)

# %% [markdown]
# Check the data dimensions for the train set and validation set (batch_size, MAX_PEP_SEQ_LEN, n_features)

# %%
print(x_train_.shape)
print(x_valid_.shape)

# %% [markdown]
# ### Flatten tensors

# %%
x_train_ = x_train_.reshape(x_train_.shape[0], -1)
x_valid_ = x_valid_.reshape(x_valid_.shape[0], -1)

# %%
batch_size = x_train_.shape[0]
n_features = x_train_.shape[1]

# %% [markdown]
# ### Make data iterable

# %%
x_train = Variable(torch.from_numpy(x_train_.astype('float32')))
y_train = Variable(torch.from_numpy(y_train_.astype('float32'))).view(-1, 1)

x_valid = Variable(torch.from_numpy(x_valid_.astype('float32')))
y_valid = Variable(torch.from_numpy(y_valid_.astype('float32'))).view(-1, 1)


# %% [markdown]
# ## Build Model

# %%
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

# %% [markdown]
# ## Select Hyper-parameters

# %%
def init_weights(m):
    """
    https://pytorch.org/docs/master/nn.init.html
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.1) # nn.init.constant_(m.bias, 0)

# %% [markdown]
# ## PCC

# %%
def pearson(pred, targ):
    p = pred - pred.mean()
    t = targ - targ.mean()
    s = p.mul(t).sum()
    s1 = p.mul(p).sum().sqrt()
    s2 = t.mul(t).sum().sqrt()
    return s / ( s1 * s2)

# %%
def error(y, y_pred):
    return 0.5*(y_pred - y)**2

# %% [markdown]
# ## Train Model

# %%
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

# %%
EPOCHS = 3000
MINI_BATCH_SIZE = 512
N_HIDDEN_NEURONS = 6
LEARNING_RATE = 0.05
PATIENCE = EPOCHS // 10    
        
net = Net(n_features, N_HIDDEN_NEURONS)

optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

net, train_loss, valid_loss, mse, pcc = train()

mse = mse.item()
pcc = pcc.item()
print(mse)
print(pcc)

# %%



