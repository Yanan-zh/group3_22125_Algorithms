# python imports
import os
import numpy as np
import math
from pprint import pprint

# here we make the PSSM Algorithm in a big function
# the input of the function should be a list contains all peptides for certain Gene
# the output is the PSSM of the certain sequence matrix

def PSSM(f_data,beta):
    (peptides, peptide_length) = ('', 9)
    # load the Alphabet
    dir_path = r'C:\Users\刘彧\Documents\GitHub\group3_22125_Algorithms'
    alphabet_file = dir_path + "\\Matrices\\alphabet"
    alphabet = np.loadtxt(alphabet_file, dtype=str)

    # load the background frequency
    bg_file = dir_path + "\Matrices\\bg.freq.fmt"
    _bg = np.loadtxt(bg_file, dtype=float)

    bg = {}
    for i in range(0, len(alphabet)):
        bg[alphabet[i]] = _bg[i]

    # load the blosum62 matrix
    blosum62_file = dir_path + "\Matrices\\blosum62.freq_rownorm"
    _blosum62 = np.loadtxt(blosum62_file, dtype=float).T

    blosum62 = {}

    for i, letter_1 in enumerate(alphabet):

        blosum62[letter_1] = {}

        for j, letter_2 in enumerate(alphabet):
            blosum62[letter_1][letter_2] = _blosum62[i, j]

    # set a function to remove the trash
    def remove_trash(raw_data):
        peptides = []
        for entries in raw_data:
            if float(entries[1]) >= 0.426:
                peptides.append(entries[0])
        return peptides

    # initialize matrix
    def initialize_matrix(peptide_length, alphabet):
        init_matrix = [0] * peptide_length

        for i in range(0, peptide_length):

            row = {}

            for letter in alphabet:
                row[letter] = 0.0

                init_matrix[i] = row

        return init_matrix

    peptides = remove_trash(f_data)

    # Amino acid count matrix
    c_matrix = initialize_matrix(peptide_length, alphabet)

    for position in range(0, peptide_length):

        for peptide in peptides:
            c_matrix[position][peptide[position]] += 1

    # Observed Frequencies Matrix (f)
    f_matrix = initialize_matrix(peptide_length, alphabet)

    for position in range(0, peptide_length):
        for letter in alphabet:
            f_matrix[position][letter] = c_matrix[position][letter] / len(peptides)

    # Pseudo Frequencies Matrix (g)
    g_matrix = initialize_matrix(peptide_length, alphabet)

    for position in range(0, peptide_length):

        for letter_1 in alphabet:
            for letter_2 in alphabet:
                g_matrix[position][letter_1] += f_matrix[position][letter_2] * blosum62[letter_1][letter_2]

    # combined matrix
    p_matrix = initialize_matrix(peptide_length, alphabet)
    r, neff = 0, 0
    for position in range(0, peptide_length):
        for letter in alphabet:
            if c_matrix[position][letter] != 0:
                r += 1
        neff += r
    neff = neff / peptide_length
    alpha = neff - 1
    for position in range(0, peptide_length):

        for a in alphabet:
            p_matrix[position][a] = (alpha * f_matrix[position][a] + beta * g_matrix[position][a]) / (alpha + beta)

    # log odd weight matrix

    w_matrix = initialize_matrix(peptide_length, alphabet)

    for position in range(0, peptide_length):

        for letter in alphabet:
            if p_matrix[position][letter] > 0:
                w_matrix[position][letter] = 2 * math.log(p_matrix[position][letter] / bg[letter]) / math.log(2)
            else:
                w_matrix[position][letter] = -999.9

    return w_matrix

# test

dir_path = r'C:\Users\刘彧\Documents\GitHub\group3_22125_Algorithms'
os.chdir(dir_path)
folder = os.getcwd()
datafile_names = os.listdir(dir_path + '\data') # make all the gene names in one list

# print(datafile_names)
test = []
for genes in datafile_names:
    test= []
    for i in range(0,5):
        f_data = np.loadtxt(dir_path + '\data\\'+ genes + '\\f00' + str(i), dtype=str).tolist()
        for j in f_data:

            test.append(j)

print(test)
a = PSSM(test,50)
print(a)