#!/home/jich/anaconda3/bin/python3
import matplotlib.pyplot as plt
import os
import pandas as pd
import sys

dir_path = r'C:\Users\Jialu\Documents\Python\22125\Project'
os.chdir(dir_path)

stafile = open('stadata.txt','r')
#infile = open('PSSM_results.txt', 'r')
tot_data = []
stata = {}

for line in stafile:
	if '#' not in line:
		line = line.strip('\n').split()
		stata[line[0]] = (int(line[1]),int(line[2]))

df = pd.read_csv('ANN_optimal_res.csv')
df = df.sort_values(by = ['AUC'],ascending=False)
df = df.iloc[0:7]

#for line in infile:
#    line = line.split()
#    data = [float(line[3]),line[0]]
#    tot_data.append(data)
#tot_data = sorted(tot_data,reverse = True)

#top7 = tot_data[:7]


#data_top_7 = []
#for i in range(7):
#    data = [stata[top7[i][1]][1],stata[top7[i][1]][0],top7[i][0],top7[i][1]]
#    data_top_7.append(data)

#names = []
#c_list = []
#n_list = []
#pcc = []

#for i in range(7):
#    names.append(data_top_7[i][3])
#    c_list.append(data_top_7[i][0])
#    n_list.append(data_top_7[i][1])
#    pcc.append('%.3f'%round(data_top_7[i][2],3))

names = df['MHC'].tolist()
c_list = [stata[i][1] for i in names]
n_list = [stata[i][0] for i in names]
pcc = df['AUC'].tolist()
pcc = ['%.3f'%round(pcc[i],3) for i in range(len(pcc))]




bar1 = plt.bar(range(7),c_list,label= 'binder',fc='r')
bar2 = plt.bar(range(7),n_list, bottom=c_list,label = 'non-binder',tick_label = names,fc = 'c')
plt.legend()
#loc='upper left'
plt.xticks(rotation=90)

for rect, label in zip(bar2,pcc):
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2, height + 800, label, ha="center", va="bottom")
plt.title('Binder/Non-binder distribution for top 7 ANN models with best AUC')
plt.savefig('ANN_AUC_bar_png')
plt.show()





#infile.close()
stafile.close()