# Import libraries
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os as os
dir_path = r'C:\Users\刘彧\Documents\GitHub\group3_22125_Algorithms'
os.chdir(dir_path)
datafile_names = os.listdir(dir_path + '\data') # make all the gene names in one list
#print((datafile_names))

for genes in datafile_names:
    B_value = []
    raw_data = np.loadtxt(dir_path + '\data\\' + genes+'\\'+genes + '.dat', dtype=str).tolist()
    for entries in raw_data:
        B_value.append(float(entries[1]))
    sns.kdeplot(B_value, shade=True)
    plt.title(genes+'\'s distribution density')
    plt.savefig(dir_path + '\plots\\' + genes + '_density.png')
    plt.close()
    plt.show()



