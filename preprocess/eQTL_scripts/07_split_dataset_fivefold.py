# split dataset: five-fold

import pandas as pd
from tqdm import tqdm
import numpy as np
pd.set_option('display.max_columns', None)

sign_path = '../../datasets/eqtl_datasets/middlefile/2_mapping_sequence/sign_prediction/'
slope_path = '../../datasets/eqtl_datasets/middlefile/2_mapping_sequence/slope_prediction/'
output_path = '../../datasets/eqtl_datasets/middlefile/4_split_fivefold/'

gtex_list = ['Heart_Left_Ventricle','Esophagus_Mucosa','Nerve_Tibial']
model_size = ['small','middle','large']

# sign_data
for bulk in gtex_list:
    print(bulk)
    for model in model_size:
        print(model)
        sign_data = pd.read_pickle(sign_path + bulk + '/' + model + '.dataset')
        spl_df = np.array_split(sign_data,5)
        for i in range(5):
            sub_df = spl_df[i].reset_index(drop=True)
            print(sub_df.shape)
            sub_df.to_pickle(output_path + 'sign_prediction/' + bulk + '/' + model + '_fold' + str(i+1) + '.dataset')

# slope_data
for bulk in gtex_list:
    print(bulk)
    for model in model_size:
        print(model)
        slope_data = pd.read_pickle(slope_path + bulk + '/' + model + '.dataset')
        spl_df = np.array_split(slope_data,5)
        for i in range(5):
            sub_df = spl_df[i].reset_index(drop=True)
            print(sub_df.shape)
            sub_df.to_pickle(output_path + 'slope_prediction/' + bulk + '/' + model + '_fold' + str(i+1) + '.dataset')