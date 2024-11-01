import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from time import time
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

onehot_dict = {
    'A': [1, 0, 0, 0],
    'T': [0, 1, 0, 0],
    'C': [0, 0, 1, 0],
    'G': [0, 0, 0, 1],
    'N': [0, 0, 0, 0],
    'a': [1, 0, 0, 0],
    't': [0, 1, 0, 0],
    'c': [0, 0, 1, 0],
    'g': [0, 0, 0, 1],
    'n': [0, 0, 0, 0]
}

def onehot_encode_sequence(sequence):
    onehot_encoded = []
    for base in sequence:
        onehot_encoded.extend(onehot_dict.get(base, [0, 0, 0, 0]))
    return np.array(onehot_encoded)

file_path = '../../datasets/benchmark_datasets/mutation_501bp.csv'
output_path = '../../datasets/benchmark_datasets/'
seq_table = pd.read_csv(file_path)
'''
               variant_id    chr ref alt       pos                                         seq_before                                          seq_after
0  chr13_43039322_A_T_b38  chr13   A   T  43039322  CATTAAAGGTAATGTGAGAGTGACTTCAGATTGTTGCTTGAAGGCA...  CATTAAAGGTAATGTGAGAGTGACTTCAGATTGTTGCTTGAAGGCA...
1   chr6_31407340_T_G_b38   chr6   T   G  31407340  atgtcattagtgttttgatagggattgcattgaatctgtagattac...  atgtcattagtgttttgatagggattgcattgaatctgtagattac...
2  chr10_47980440_C_T_b38  chr10   C   T  47980440  gaaatcacaatgaaaattagaaaattatttgaaatgaatgaaaata...  gaaatcacaatgaaaattagaaaattatttgaaatgaatgaaaata...
3   chr17_8106013_A_C_b38  chr17   A   C   8106013  tggatgcagtgatgtggtggtgcatgcctgtagtcccagctaccag...  tggatgcagtgatgtggtggtgcatgcctgtagtcccagctaccag...
4   chr6_37636844_C_T_b38   chr6   C   T  37636844  TTGTAGGGTGGAGTGGGGAAGTTATCAGTCTTCAGACTGCCCAGGC...  TTGTAGGGTGGAGTGGGGAAGTTATCAGTCTTCAGACTGCCCAGGC...
'''

seq_table['time_before'] = 0
seq_table['time_after'] = 0
seq_table['embedding_before'] = 0
seq_table['embedding_before'] = seq_table['embedding_before'].astype('object')
seq_table['embedding_after'] = 0
seq_table['embedding_after'] = seq_table['embedding_after'].astype('object')

for i in range(len(seq_table)):
    seq_before = seq_table['seq_before'][i]
    seq_after = seq_table['seq_after'][i]
    variant_id = seq_table['variant_id'][i]

    t1 = time()
    new1 = onehot_encode_sequence(seq_before)
    t2 = time()
    time1 = t2 - t1

    t1 = time()
    new2 = onehot_encode_sequence(seq_after)
    t2 = time()
    time2 = t2 - t1

    seq_table['time_before'][i] = time1
    seq_table['time_after'][i] = time2
    seq_table['embedding_before'][i] = new1
    seq_table['embedding_after'][i] = new2

print(seq_table.head())
seq_table.to_pickle(output_path + 'onehot.dataset')
'''
               variant_id    chr ref alt       pos  \
0  chr13_43039322_A_T_b38  chr13   A   T  43039322   
1   chr6_31407340_T_G_b38   chr6   T   G  31407340   
2  chr10_47980440_C_T_b38  chr10   C   T  47980440   
3   chr17_8106013_A_C_b38  chr17   A   C   8106013
4   chr6_37636844_C_T_b38   chr6   C   T  37636844

                                          seq_before  \
0  CATTAAAGGTAATGTGAGAGTGACTTCAGATTGTTGCTTGAAGGCA...
1  atgtcattagtgttttgatagggattgcattgaatctgtagattac...
2  gaaatcacaatgaaaattagaaaattatttgaaatgaatgaaaata...
3  tggatgcagtgatgtggtggtgcatgcctgtagtcccagctaccag...
4  TTGTAGGGTGGAGTGGGGAAGTTATCAGTCTTCAGACTGCCCAGGC...

                                           seq_after  time_before  time_after  \
0  CATTAAAGGTAATGTGAGAGTGACTTCAGATTGTTGCTTGAAGGCA...     0.000998         0.0
1  atgtcattagtgttttgatagggattgcattgaatctgtagattac...     0.000994         0.0
2  gaaatcacaatgaaaattagaaaattatttgaaatgaatgaaaata...     0.000000         0.0
3  tggatgcagtgatgtggtggtgcatgcctgtagtcccagctaccag...     0.000000         0.0
4  TTGTAGGGTGGAGTGGGGAAGTTATCAGTCTTCAGACTGCCCAGGC...     0.000000         0.0

                                    embedding_before  \
0  [0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, ...
1  [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, ...
2  [0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, ...
3  [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, ...
4  [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, ...

                                     embedding_after
0  [0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, ...
1  [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, ...
2  [0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, ...
3  [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, ...
4  [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, ...
'''