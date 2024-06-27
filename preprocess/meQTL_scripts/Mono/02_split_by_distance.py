import os
import pandas as pd
import numpy as np
import subprocess
from tqdm import tqdm
import random
import json
import warnings
warnings.filterwarnings('ignore')

cross_path = '../../../datasets/middlefile/Mono_meQTL_annotation/'
save_path = '../../../datasets/labeled_Mono/'

'''
cross_table:
        CpG             SNP      Beta Ref Alt   CHR   CpG_POS   SNP_POS label
0  cg15715337  1:85591599_G_A -6.415984   G   A  chr1  85600447  85591599     0
1  cg15715337  1:85592937_C_T -6.401692   C   T  chr1  85600447  85592937     0
2  cg15715337  1:85592946_C_T -6.399971   C   T  chr1  85600447  85592946     0
3  cg15715337  1:85593116_C_T -6.393680   C   T  chr1  85600447  85593116     0
4  cg25835058  1:15404145_A_G -3.164003   A   G  chr1  15407757  15404145     0
'''
'''
cross_table with distance:
         CpG             SNP      Beta Ref Alt   CHR   CpG_POS   SNP_POS label  distance
0  cg15715337  1:85591599_G_A -6.415984   G   A  chr1  85600447  85591599     0      8848
1  cg15715337  1:85592937_C_T -6.401692   C   T  chr1  85600447  85592937     0      7510
2  cg15715337  1:85592946_C_T -6.399971   C   T  chr1  85600447  85592946     0      7501
3  cg15715337  1:85593116_C_T -6.393680   C   T  chr1  85600447  85593116     0      7331
4  cg25835058  1:15404145_A_G -3.164003   A   G  chr1  15407757  15404145     0      3612
'''

# add distance
for i in range(22):
	chr = 'chr' + str(i+1)
	cross_table = pd.read_pickle(cross_path + chr + '.pkl')
	cross_table['distance'] = 0
	for i in tqdm(range(cross_table.shape[0])):
		cpg_pos = cross_table['CpG_POS'][i]
		snp_pos = cross_table['SNP_POS'][i]
		distance = np.abs(int(cpg_pos) - int(snp_pos))
		cross_table['distance'][i] = distance
		cross_table.to_pickle(cross_path + chr + '_distance.pkl')
	print(cross_table.head())   


# split by distance
cross_path = '../../../datasets/middlefile/Mono_meQTL_annotation/'
save_path = '../../../datasets/labeled_Mono/'
for i in range(22):
	chr = 'chr' + str(i+1)
	cross_table = pd.read_pickle(cross_path + chr + '_distance.pkl')
    # small table
	len_sample = cross_table[cross_table['distance'] <= 10_000].reset_index(drop=True)
	len_sample.to_pickle(save_path + chr + '_small.dataset')
	# large table
	len_sample = cross_table[(cross_table['distance'] > 10_000) & (cross_table['distance'] <= 100_000)].reset_index(drop=True)
	len_sample.to_pickle(save_path + chr + '_large.dataset')

# down-sampling
cross_path = '../../../datasets/labeled_Mono/'
save_path = '../../../datasets/sampled_Mono/'
for i in range(22):
	chr = 'chr' + str(i+1)
	# small table
	cross_table = pd.read_pickle(cross_path + chr + '_small.dataset')
	min_sample = min(cross_table['label'].value_counts()[0], cross_table['label'].value_counts()[1])
	up_table = cross_table[cross_table['label']==1].sample(frac=1,replace=False).reset_index(drop=True)[0:min_sample]
	down_table = cross_table[cross_table['label']==0].sample(frac=1,replace=False).reset_index(drop=True)[0:min_sample]
	new_table = pd.concat([up_table,down_table]).reset_index(drop=True)
	#print(new_table['label'].value_counts())
	new_table.to_pickle(save_path + chr + '_small.dataset')
	# large table
	cross_table = pd.read_pickle(cross_path + chr + '_large.dataset')
	min_sample = min(cross_table['label'].value_counts()[0], cross_table['label'].value_counts()[1])
	min_sample = int(min_sample/4)
	up_table = cross_table[cross_table['label']==1].sample(frac=1,replace=False).reset_index(drop=True)[0:min_sample]
	down_table = cross_table[cross_table['label']==0].sample(frac=1,replace=False).reset_index(drop=True)[0:min_sample]
	new_table = pd.concat([up_table,down_table]).reset_index(drop=True)
	#print(new_table['label'].value_counts())
	new_table.to_pickle(save_path + chr + '_large.dataset')