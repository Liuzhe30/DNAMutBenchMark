import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from time import time
import gpn.model
from transformers import AutoTokenizer, AutoModel
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

model_path = "songlab/gpn-brassicales"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path).to(device)

cell_list = ['CD4', 'mono']
model_size = {'20001': 'small', '200001': 'large'}
model_cutting = {'small': 19, 'large': 199}


def split_seq(sequence):
    seq_list = []
    length = len(sequence)
    model = model_size[str(int(length))]
    n = model_cutting[model]

    # first 250bp
    seq_list.append(sequence[0:250])
    # first n*500bp
    for i in range(n):
        seq_list.append(sequence[250 + i * 500:250 + (i + 1) * 500])
    # mutation part
    seq_list.append(sequence[250 + 500 * n:250 + 500 * n + 501])
    # second n*500bp
    for i in range(n):
        seq_list.append(sequence[250 + 500 * n + 501 + i * 500:250 + 500 * n + 501 + (i + 1) * 500])
    # second 250bp
    seq_list.append(sequence[250 + 500 * n + 501 + 500 * n:250 + 500 * n + 501 + 500 * n + 250])

    return seq_list


def gpn_embedding(seq_list):
    embedding_list = []
    for sequence in seq_list:
        input_ids = tokenizer(sequence, return_tensors='pt', return_attention_mask=False, return_token_type_ids=False)[
            "input_ids"]
        input_ids = input_ids.to(device)
        model.eval()
        with torch.no_grad():
            embedding = model(input_ids=input_ids).last_hidden_state  # [1, sequence_length, 512]
        new = embedding[0].data.cpu().numpy()
        embedding_list.append(new)
    embedding = np.array(embedding_list[0])
    for item in embedding_list[1:]:
        embedding = np.concatenate([embedding, item], axis=0)
    avg_embedding = np.mean(embedding, axis=0)  # [512, ]
    return avg_embedding


file_path = '../../datasets/benchmark_meqtl_dataset/slope_prediction/'
output_path = '../../datasets_embedding/gpn/meqtl_datasets/slope_prediction/'

for tissue in cell_list:
    for s in ['train', 'valid', 'test']:
        for m in model_cutting.keys():
            data = pd.read_pickle(file_path + '/' + tissue + '/' + m + '_' + s + '.dataset')
            data['gpn_before'] = 0
            data['gpn_before_time'] = 0
            data['gpn_before'] = data['gpn_before'].astype('object')
            data['gpn_after'] = 0
            data['gpn_after_time'] = 0
            data['gpn_after'] = data['gpn_after'].astype('object')
            for i in range(len(data)):
                seq_before = data['seq_before'][i]
                sequence1_list = split_seq(seq_before)
                t1 = time()
                gpn1 = gpn_embedding(sequence1_list)  # (x,512)
                t2 = time()
                data['gpn_before_time'][i] = t2 - t1
                data['gpn_before'][i] = gpn1
                seq_after = data['seq_after'][i]
                sequence2_list = split_seq(seq_after)
                t1 = time()
                gpn2 = gpn_embedding(sequence2_list)
                t2 = time()
                data['gpn_after_time'][i] = t2 - t1
                data['gpn_after'][i] = gpn2
                # print(data.head())
            if not os.path.exists(output_path + tissue + '/'):
                os.makedirs(output_path + tissue + '/')
            print(output_path + tissue + '/' + m + '_' + s + '.dataset')
            data.to_pickle(output_path + tissue + '/' + m + '_' + s + '.dataset')