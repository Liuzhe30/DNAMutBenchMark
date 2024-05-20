# post-process after mapping DNA sequences
import pandas as pd
from tqdm import tqdm
pd.set_option('display.max_columns', None)

file_path = '../../datasets/eqtl_datasets/middlefile/2_mapping_sequence/sign_prediction/'
output_path = '../../datasets/eqtl_datasets/middlefile/2_mapping_sequence_post/sign_prediction/'

model_size = {'small':1_000,'middle':10_000,'large':100_000}
gtex_list = ['Heart_Left_Ventricle','Esophagus_Mucosa','Nerve_Tibial']

for bulk in gtex_list:
    for model in model_size.keys():
        data = pd.read_pickle(file_path + bulk + '/' + model + '.dataset')
        data_check = pd.read_pickle(file_path + bulk + '/' + model + '.dataset')
        max_range = model_size[model]
        for i in range(len(data_check)):
            seq_len = data_check['seq_len'][i]
            variant_id = data_check['variant_id'][i]
            slope = data_check['slope'][i]
            if(seq_len != max_range * 2 + 1):
                data = data.drop(data[(data['variant_id']==variant_id)&(data['slope']==slope)].index)
        data = data.reset_index(drop=True)
        data.to_pickle(output_path + bulk + '/' + model + '.dataset')

file_path = '../../datasets/eqtl_datasets/middlefile/2_mapping_sequence/slope_prediction/'
output_path = '../../datasets/eqtl_datasets/middlefile/2_mapping_sequence_post/slope_prediction/'

model_size = {'small':1_000,'middle':10_000,'large':100_000}
gtex_list = ['Heart_Left_Ventricle','Esophagus_Mucosa','Nerve_Tibial']

for bulk in gtex_list:
    for model in model_size.keys():
        data = pd.read_pickle(file_path + bulk + '/' + model + '.dataset')
        data_check = pd.read_pickle(file_path + bulk + '/' + model + '.dataset')
        max_range = model_size[model]
        for i in range(len(data_check)):
            seq_len = data_check['seq_len'][i]
            variant_id = data_check['variant_id'][i]
            slope = data_check['slope'][i]
            if(seq_len != max_range * 2 + 1):
                data = data.drop(data[(data['variant_id']==variant_id)&(data['slope']==slope)].index)
        data = data.reset_index(drop=True)
        data.to_pickle(output_path + bulk + '/' + model + '.dataset')