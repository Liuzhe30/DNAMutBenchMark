import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from time import time

# https://github.com/google-deepmind/deepmind-research/blob/master/enformer/README.md, one hot encoded in order 'ACGT'
eyes = np.eye(4)
gene_dict = {'A':eyes[0], 'C':eyes[1], 'G':eyes[2], 'T':eyes[3], 'N':np.zeros(4),
             'a':eyes[0], 'c':eyes[1], 'g':eyes[2], 't':eyes[3], 'n':np.zeros(4)
             }

enformer = hub.load("https://www.kaggle.com/models/deepmind/enformer/frameworks/TensorFlow2/variations/enformer/versions/1").model
print('test load successful!')
maxlen = 393216
fasta_path = '../../datasets/raw/chr_fasta_hg38/'

def mutation_center_seq(variant_id):
    chr_str = variant_id.split('_')[0]
    position = int(variant_id.split('_')[1])
    before_mutation = variant_id.split('_')[2]
    after_mutation = variant_id.split('_')[3]
    with open(fasta_path + chr_str + '_new.fasta') as fa:
        line = fa.readline()        
        range_seq = int(maxlen/2)       
        sequence_before = line[position - range_seq:position + range_seq]
        if(line[position - 1] >= 'a' and line[position - 1] <= 'z'):
            sequence_after = line[position - range_seq: position - 1] + after_mutation.lower() + line[position: position + range_seq]
        else:
            sequence_after = line[position - range_seq: position - 1] + after_mutation + line[position: position + range_seq]
    return sequence_before, sequence_after

def fetch_enformer_results(sequence):
    seq_list = []
    for strr in sequence:
        seq_list.append(gene_dict[strr])
    seq_array = np.array(seq_list)
    tensor = tf.convert_to_tensor(seq_array, tf.float32)
    tensor = tf.expand_dims(tensor, axis=0)
    result = np.array(enformer.predict_on_batch(tensor)['human'])
    return result

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

    sequence_before, sequence_after = mutation_center_seq(variant_id)
    if(len(sequence_before) != maxlen or len(sequence_after) != maxlen):
        continue

    t1 = time()
    new1 = fetch_enformer_results(sequence_before)
    t2 = time()
    time1 = t2 - t1

    t1 = time()
    new2 = fetch_enformer_results(sequence_after)
    t2 = time()
    time2 = t2 - t1

    seq_table['time_before'][i] = time1
    seq_table['time_after'][i] = time2
    seq_table['embedding_before'][i] = new1
    seq_table['embedding_after'][i] = new2

print(seq_table.head())
seq_table.to_pickle(output_path + 'enformer.dataset')