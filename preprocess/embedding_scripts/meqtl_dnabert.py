# fetch DNABERT embedding: -> [x,768]
# https://github.com/MAGICS-LAB/DNABERT_2
import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from time import time
from transformers import AutoTokenizer, AutoModel
import warnings
warnings.filterwarnings('ignore')

tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)

cell_list = ['CD4','mono']
model_size = {'20001':'small','200001':'large'}
model_cutting = {'small':19,'large':199}

def split_seq(sequence):
    seq_list = []
    length = len(sequence)
    model = model_size[str(int(length))]
    n = model_cutting[model]

    # first 250bp
    seq_list.append(sequence[0:250])
    # first n*500bp
    for i in range(n):
        seq_list.append(sequence[250+i*500:250+(i+1)*500])
    # mutation part
    seq_list.append(sequence[250+500*n:250+500*n+501])
    # second n*500bp
    for i in range(n):
        seq_list.append(sequence[250+500*n+501+i*500:250+500*n+501+(i+1)*500])
    # second 250bp
    seq_list.append(sequence[250+500*n+501+500*n:250+500*n+501+500*n+250])
    
    return seq_list

def dnabert_embedding(seq_list):
    embedding_list = []
    for sequence in seq_list:
        inputs = tokenizer(sequence, return_tensors = 'pt')["input_ids"]
        hidden_states = model(inputs)[0] # [1, sequence_length, 768]
        new = hidden_states.data.cpu().numpy()
        new = new.reshape([new.shape[1],new.shape[2]])
        embedding_list.append(new)
    embedding = np.array(embedding_list[0])
    for item in embedding_list[1:]:
        print(item.shape)
        embedding = np.concatenate([embedding,item],axis=0)
    avg_embedding = np.mean(embedding, axis=0) # (1,768)
    return avg_embedding

file_path = '../../datasets/benchmark_meqtl_dataset/slope_prediction/'
output_path = '../../datasets_embedding/dnabert2/meqtl_datasets/slope_prediction/'

for tissue in cell_list:
    for s in ['train','valid','test']:
        for m in model_cutting.keys():
            data = pd.read_pickle(file_path + '/' + tissue + '/' + m + '_' + s + '.dataset')
            data['dnabert_before'] = 0
            data['dnabert_before_time'] = 0
            data['dnabert_before'] = data['dnabert_before'].astype('object')
            data['dnabert_after'] = 0
            data['dnabert_after_time'] = 0
            data['dnabert_after'] = data['dnabert_after'].astype('object')
            for i in range(len(data)):
                seq_before = data['seq_before'][i]
                sequence1_list = split_seq(seq_before)
                t1 = time()
                dnabert1 = dnabert_embedding(sequence1_list) # # (1,768)
                t2 = time()
                data['dnabert_before_time'][i] = t2 - t1
                data['dnabert_before'][i] = dnabert1
                seq_after = data['seq_after'][i]
                sequence2_list = split_seq(seq_after)
                t1 = time()
                dnabert2 = dnabert_embedding(sequence2_list)
                t2 = time()
                data['dnabert_after_time'][i] = t2 - t1
                data['dnabert_after'][i] = dnabert2
                #print(data.head())
            data.to_pickle(output_path + '/' + tissue + '/' + m + '_' + s + '.dataset')