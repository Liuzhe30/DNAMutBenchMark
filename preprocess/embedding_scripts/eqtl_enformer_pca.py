import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

compare_tissue_list = ['Esophagus_Mucosa','Heart_Left_Ventricle','Nerve_Tibial']
model_size = {'2001':'small','20001':'middle','200001':'large'}
model_cutting = {'small':1,'middle':19,'large':199}

output_path = '../../datasets_embedding/enformer_pca/eqtl_datasets/sign_prediction/'
# sign prediction
for tissue in compare_tissue_list:
    for s in ['train','valid','test']:
        for m in model_cutting.keys():
            data = pd.read_pickle('../../datasets_embedding/enformer/eqtl_datasets/sign_prediction/' + tissue + '/' + m + '_' + s + '.dataset')
            new_df = pd.DataFrame(columns=['variant_id', 'label', 'enformer_pca_before', 'enformer_pca_after'])
            for i in range(len(data)):
                result_before = data['result_before'][i]
                pca = PCA(n_components=10)
                enformer_pca_before = pca.fit_transform(result_before) 

                result_after = data['result_after'][i]
                pca = PCA(n_components=10)
                enformer_pca_after = pca.fit_transform(result_after) 

                new_df = new_df._append([{'variant_id': data['variant_id'][i], 'label': data['label'][i], 'enformer_pca_before': result_before, 
                                        'enformer_pca_after': result_after}], ignore_index=True)
            
            new_df.to_pickle(output_path + '/' + tissue + '/' + m + '_' + s + '.dataset')

output_path = '../../datasets_embedding/enformer_pca/eqtl_datasets/slope_prediction/'
# slope prediction
for tissue in compare_tissue_list:
    for s in ['train','valid','test']:
        for m in model_cutting.keys():
            data = pd.read_pickle('../../datasets_embedding/enformer/eqtl_datasets/slope_prediction/' + tissue + '/' + m + '_' + s + '.dataset')
            new_df = pd.DataFrame(columns=['variant_id', 'slope', 'enformer_pca_before', 'enformer_pca_after'])
            for i in range(len(data)):
                result_before = data['result_before'][i]
                pca = PCA(n_components=10)
                enformer_pca_before = pca.fit_transform(result_before) 

                result_after = data['result_after'][i]
                pca = PCA(n_components=10)
                enformer_pca_after = pca.fit_transform(result_after) 

                new_df = new_df._append([{'variant_id': data['variant_id'][i], 'slope': data['slope'][i], 'enformer_pca_before': result_before, 
                                        'enformer_pca_after': result_after}], ignore_index=True)
            
            new_df.to_pickle(output_path + '/' + tissue + '/' + m + '_' + s + '.dataset')

