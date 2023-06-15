# in this section we are going to seperate the whole data in to combined and not combined sequence
import numpy as np
import os
import matplotlib.pyplot as plt
dir_path = r'C:\Users\刘彧\Documents\GitHub\group3_22125_Algorithms'
os.chdir(dir_path)
datafile_names = os.listdir(dir_path + '\data') # make all the gene names in one list
#print(len(datafile_names))
c_list,n_list= [],[]

for genes in datafile_names:
    c_count,n_count = 0,0
    raw_data = np.loadtxt(dir_path + '\data\\' + genes+'\\'+genes + '.dat', dtype=str).tolist()
    for peptide in raw_data:
        if float(peptide[1]) >= 0.426:
            c_count += 1
        else:
            n_count += 1
    c_list.append(c_count)
    n_list.append(n_count)
print(datafile_names)
print(c_list)
print(n_list)
print(len(datafile_names))
print(len(c_list))
print(len(n_list))

# make the bar plot
plt.bar(range(35),c_list,label= 'binder',fc='r')
plt.bar(range(35),n_list, bottom=c_list,label = 'non-binder',tick_label = datafile_names,fc = 'c')
plt.legend()
plt.xticks(rotation=90)
plt.title('binder and Non-binder distribution for all genes')
plt.savefig(dir_path + '\plots\\'+'bar_png')
plt.show()