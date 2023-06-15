# in this section we are going to seperate the whole data in to combined and not combined sequence
import numpy as np
import os
from Bio import motifs
from Bio.Seq import Seq
import matplotlib.pyplot as plt
dir_path = '/mnt/c/Users/刘彧/Documents/GitHub/group3_22125_Algorithms/'
# os.chdir(dir_path)
#print(os.listdir('/mnt/c/Users/刘彧/Documents/GitHub/group3_22125_Algorithms/data'))
datafile_names = os.listdir(os.path.join(dir_path, 'data')) # make all the gene names in one list
#print((datafile_names))
c_list,n_list= [],[]

for genes in datafile_names:
    c_list, n_list = [], []
    raw_data = np.loadtxt(dir_path + '/data/' + genes+'/'+genes + '.dat', dtype=str).tolist()
    for peptide in raw_data:
        if float(peptide[1]) >= 0.426:
            c_list.append(peptide[0])
        else:
            n_list.append(peptide[0])
# print(c_list)
# print(n_list)
motif = motifs.create(c_list)
print(motif)