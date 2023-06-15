# # in this section we are going to seperate the whole data in to combined and not combined sequence
# import numpy as np
# import os
# from Bio import motifs
# from Bio.Seq import Seq
# import matplotlib.pyplot as plt
# dir_path = '/mnt/c/Users/刘彧/Documents/GitHub/group3_22125_Algorithms/'
# # os.chdir(dir_path)
# #print(os.listdir('/mnt/c/Users/刘彧/Documents/GitHub/group3_22125_Algorithms/data'))
# datafile_names = os.listdir(os.path.join(dir_path, 'data')) # make all the gene names in one list
# #print((datafile_names))
# c_list,n_list= [],[]
#
# for genes in datafile_names:
#     c_list, n_list = [], []
#     raw_data = np.loadtxt(dir_path + '/data/' + genes+'/'+genes + '.dat', dtype=str).tolist()
#     for peptide in raw_data:
#         if float(peptide[1]) >= 0.426:
#             c_list.append(peptide[0])
#         else:
#             n_list.append(peptide[0])
# print(c_list)
# print(n_list)
# motif = motifs.create(sequences, alphabet='ACDEFGHIKLMNPQRSTVWY')
# print(motif)



# from Bio import motifs
# from Bio.Seq import Seq
# import matplotlib.pyplot as plt
#
# # 定义嵌套的氨基酸序列
# nested_sequences = [["A", "R", "N", "D", "C"],
#                     ["E", "Q", "G", "H", "I"],
#                     ["L", "K", "M", "F", "P"],
#                     ["S", "T", "W", "Y", "V"]]
#
# # nested_sequences = [["A", "C", "T", "G", "C"],
# #                     ["C", "T", "G", "A", "C"],
# #                     ["A", "C", "C", "G", "T"],
# #                     ["A", "C", "T", "G", "C"]]
#
# # 将嵌套列表转换为单个字符串
# sequences = ["".join(sublist) for sublist in nested_sequences]
#
# # 创建Motif对象
# motif = motifs.create(sequences, alphabet='ACDEFGHIKLMNPQRSTVWY')
#
# # 绘制Logo图
# motif.weblogo('seq.html',version=3,fmt='pdf')

# # 显示Logo图
# plt.rcParams["figure.figsize"] = (8, 4)  # 设置图形大小
# motif_logo.draw()
# plt.title("Amino Acid Logo")  # 设置标题
# plt.ylabel("Bits")  # 设置y轴标签
# plt.show()
