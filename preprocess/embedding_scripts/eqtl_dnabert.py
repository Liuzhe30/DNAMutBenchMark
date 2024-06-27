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

gtex_list = ['Heart_Left_Ventricle','Esophagus_Mucosa','Nerve_Tibial']
model_size = {'2001':'small','20001':'middle','200001':'large'}
model_cutting = {'small':1,'middle':19,'large':199}

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
        embedding_mean = torch.mean(hidden_states[0], dim=0)
        new = embedding_mean.data.cpu().numpy()
        new = new.reshape([1,new.shape[0]])
        embedding_list.append(new)
    embedding = np.array(embedding_list[0])
    for item in embedding_list[1:]:
        embedding = np.concatenate([embedding,item],axis=0)
    return embedding

# sign prediction - split by chr
file_path = '../../datasets/benchmark_eqtl_dataset/sign_prediction/'
output_path = '../../datasets_embedding/dnabert2/eqtl_datasets/sign_prediction/'

for tissue in gtex_list:
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
                dnabert1 = dnabert_embedding(sequence1_list) # (x,768)
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
            data.to_pickle(output_path + '/' + tissue + '/' + m + '_' + s + '.dataset')

# slope prediction - split by chr
file_path = '../../datasets/benchmark_eqtl_dataset/slope_prediction/'
output_path = '../../datasets_embedding/dnabert2/eqtl_datasets/slope_prediction/'

for tissue in gtex_list:
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
                dnabert1 = dnabert_embedding(sequence1_list) # (x,768)
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
                print(data.head())
            data.to_pickle(output_path + '/' + tissue + '/' + m + '_' + s + '.dataset')