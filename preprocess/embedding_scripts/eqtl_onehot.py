import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from time import time
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

gtex_list = ['Heart_Left_Ventricle', 'Esophagus_Mucosa', 'Nerve_Tibial']
model_size = {'2001': 'small', '20001': 'middle', '200001': 'large'}
model_cutting = {'small': 1, 'middle': 19, 'large': 199}

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
    # 初始化一个空列表用于存储One-Hot编码
    onehot_encoded = []
    # 遍历序列中的每个碱基
    for base in sequence:
        onehot_encoded.extend(onehot_dict.get(base, [0, 0, 0, 0]))
    # 转换为NumPy数组
    return np.array(onehot_encoded)

# slope prediction - split by chr
file_path = '../../datasets/benchmark_eqtl_dataset/sign_prediction/'
output_path = '../../datasets_embedding/onehot/eqtl_datasets/sign_prediction/'

for tissue in gtex_list:
    for s in ['train', 'valid', 'test']:
        for m in model_cutting.keys():
            data = pd.read_pickle(file_path + '/' + tissue + '/' + m + '_' + s + '.dataset')
            data['onehot_before'] = 0
            data['onehot_before_time'] = 0
            data['onehot_before'] = data['onehot_before'].astype('object')
            data['onehot_after'] = 0
            data['onehot_after_time'] = 0
            data['onehot_after'] = data['onehot_after'].astype('object')
            for i in range(len(data)):
                seq_before = data['seq_before'][i]
                t1 = time()
                onehot1 = onehot_encode_sequence(seq_before)  # (x,512)
                t2 = time()
                data['onehot_before_time'][i] = t2 - t1
                data['onehot_before'][i] = onehot1
                seq_after = data['seq_after'][i]
                t1 = time()
                onehot2 = onehot_encode_sequence(seq_after)
                t2 = time()
                data['onehot_after_time'][i] = t2 - t1
                data['onehot_after'][i] = onehot2
                # print(data.head())
            if not os.path.exists(output_path + tissue + '/'):
                os.makedirs(output_path + tissue + '/')
            print(output_path + tissue + '/' + m + '_' + s + '.dataset')
            data.to_pickle(output_path + tissue + '/' + m + '_' + s + '.dataset')
