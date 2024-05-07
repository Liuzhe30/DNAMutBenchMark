# split dataset by chr
# training: chr1-9,chr14-22
# validation: chr12-13
# test: chr10-11

import pandas as pd
from pandas import read_parquet
from tqdm import tqdm
pd.set_option('display.max_columns', None)

sign_path = '../../datasets/eqtl_datasets/middlefile/2_mapping_sequence/sign_prediction/'
slope_path = '../../datasets/eqtl_datasets/middlefile/2_mapping_sequence/slope_prediction/'
output_path = '../../datasets/eqtl_datasets/middlefile/3_split_by_chr/'

gtex_list = ['Heart_Left_Ventricle','Esophagus_Mucosa','Nerve_Tibial']
model_size = ['small','middle','large']

# sign_data
for bulk in gtex_list:
    print(bulk)
    for model in model_size:
        print(model)
        train_data = pd.DataFrame()
        valid_data = pd.DataFrame()
        test_data = pd.DataFrame()
        sign_data = pd.read_pickle(sign_path + bulk + '/' + model + '.dataset')
        for i in range(len(sign_data)): 
            variant_id = sign_data['variant_id'].values[i]
            chr = int(variant_id.split('_')[0][3:])
            if((chr >= 1 and chr <= 9) or (chr >= 14 and chr <= 22)):
                train_data = train_data._append(sign_data[i:i+1], ignore_index=True)
            elif(chr >= 12 and chr <= 13):
                valid_data = valid_data._append(sign_data[i:i+1], ignore_index=True)
            else:
                test_data = test_data._append(sign_data[i:i+1], ignore_index=True)
        train_data = train_data.reset_index(drop=True)
        train_data.to_pickle(output_path + 'sign_prediction/' + bulk + '/' + model + '_train.dataset')
        print(train_data.shape)
        valid_data = valid_data.reset_index(drop=True)
        valid_data.to_pickle(output_path + 'sign_prediction/' + bulk + '/' + model + '_valid.dataset')
        print(valid_data.shape)
        test_data = test_data.reset_index(drop=True)
        test_data.to_pickle(output_path + 'sign_prediction/' + bulk + '/' + model + '_test.dataset')
        print(test_data.shape)


# slope_data
for bulk in gtex_list:
    print(bulk)
    for model in model_size:
        print(model)
        train_data = pd.DataFrame()
        valid_data = pd.DataFrame()
        test_data = pd.DataFrame()
        slope_data = pd.read_pickle(slope_path + bulk + '/' + model + '.dataset')
        for i in range(len(slope_data)): 
            variant_id = slope_data['variant_id'].values[i]
            chr = int(variant_id.split('_')[0][3:])
            if((chr >= 1 and chr <= 9) or (chr >= 14 and chr <= 22)):
                train_data = train_data._append(slope_data[i:i+1], ignore_index=True)
            elif(chr >= 12 and chr <= 13):
                valid_data = valid_data._append(slope_data[i:i+1], ignore_index=True)
            else:
                test_data = test_data._append(slope_data[i:i+1], ignore_index=True)
        train_data = train_data.reset_index(drop=True)
        train_data.to_pickle(output_path + 'slope_prediction/' + bulk + '/' + model + '_train.dataset')
        print(train_data.shape)
        valid_data = valid_data.reset_index(drop=True)
        valid_data.to_pickle(output_path + 'slope_prediction/' + bulk + '/' + model + '_valid.dataset')
        print(valid_data.shape)
        test_data = test_data.reset_index(drop=True)
        test_data.to_pickle(output_path + 'slope_prediction/' + bulk + '/' + model + '_test.dataset')
        print(test_data.shape)
