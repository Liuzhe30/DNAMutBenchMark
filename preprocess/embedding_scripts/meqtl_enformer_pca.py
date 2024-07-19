import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

model_size = {'small':10_000,'large':100_000}
compare_tissue_list = ['CD4','mono']

# slope prediction
output_path = '../../datasets_embedding/enformer_pca/meqtl_datasets/slope_prediction/'
for m in model_size.keys():
    for s in ['train','valid','test']:
        for tissue in compare_tissue_list:
            
            data = pd.read_pickle('../../datasets_embedding/enformer/meqtl_datasets/slope_prediction/' + tissue + '/' + m + '_' + s + '.dataset')
            new_df = pd.DataFrame(columns=['variant_id', 'Beta', 'enformer_pca_before', 'enformer_pca_after'])
            for i in range(len(data)):
                result_before = np.array(data['result_before'][i])
                result_before = result_before.reshape([result_before.shape[1],result_before.shape[-1]])
                pca = PCA(n_components=10)
                enformer_pca_before = pca.fit_transform(result_before) 

                result_after = np.array(data['result_after'][i])
                result_after = result_after.reshape([result_after.shape[1],result_after.shape[-1]])
                pca = PCA(n_components=10)
                enformer_pca_after = pca.fit_transform(result_after) 

                new_df = new_df._append([{'variant_id': data['variant_id'][i], 'Beta': data['Beta'][i], 'enformer_pca_before': enformer_pca_before, 
                                        'enformer_pca_after': enformer_pca_after}], ignore_index=True)
            
            new_df.to_pickle(output_path + '/' + tissue + '/' + m + '_' + s + '.dataset')
