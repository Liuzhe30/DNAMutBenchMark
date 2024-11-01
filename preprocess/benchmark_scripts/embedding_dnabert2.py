# fetch DNABERT embedding: -> [x,768]
# https://github.com/MAGICS-LAB/DNABERT_2
import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from time import time
from transformers import AutoTokenizer, AutoModel
pd.set_option('display.max_columns', None)
import warnings
warnings.filterwarnings('ignore')

tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)

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
    inputs1 = tokenizer(seq_before, return_tensors = 'pt')["input_ids"]
    hidden_states1 = model(inputs1)[0] # [1, sequence_length, 768]
    new1 = hidden_states1.data.cpu().numpy()
    t2 = time()
    time1 = t2 - t1

    t1 = time()
    inputs2 = tokenizer(seq_after, return_tensors = 'pt')["input_ids"]
    hidden_states2 = model(inputs2)[0] # [1, sequence_length, 768]
    new2 = hidden_states2.data.cpu().numpy()
    t2 = time()
    time2 = t2 - t1

    seq_table['time_before'][i] = time1
    seq_table['time_after'][i] = time2
    seq_table['embedding_before'][i] = new1
    seq_table['embedding_after'][i] = new2

print(seq_table.head())
seq_table.to_pickle(output_path + 'dnabert.dataset')
    