import os
import pandas as pd
import numpy as np
import subprocess
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# step 1 CpG and SNP annotation to methylation position of meQTLs
#'''
meqtl_path = '../../../datasets/raw/gwama_cis_random_Mono_n2/gwama_cis_random_Mono_'
cpg_annotation_path = '../../../datasets/middlefile/clean_epic/'
save_path = '../../../datasets/middlefile/Mono_meQTL_annotation/'

sum = 0
for i in range(22):
	chr = 'chr' + str(i+1)
	main_table = pd.read_csv(meqtl_path + chr + '_n2.out', sep='\t')
	cpg_table = pd.read_pickle(cpg_annotation_path + 'cpg_' + chr + '.pkl')
	indexed_cpg = cpg_table.set_index('IlmnID')

	new_table = pd.DataFrame(columns=['CpG', 'SNP', 'Beta', 'Ref', 'Alt', 'CHR', 'CpG_POS', 'SNP_POS','label'])
	for i in tqdm(range(main_table.shape[0])): 
		CpG_CHR = 'space'
		SNP_CHR = 'space'
		try:
			# CpG annotation
			cpg = main_table['CpG'][i]
			CpG_CHR = indexed_cpg[indexed_cpg.index==cpg]['CHR'].values[0]
			CpG_POS = indexed_cpg[indexed_cpg.index==cpg]['CpG_pos'].values[0]
			# SNP annotation
			snp = main_table['SNP'][i]
			SNP_CHR = 'chr' + snp.split(':')[0]
			SNP_POS = int(snp.split(':')[1].split('_')[0])
			SNP_check_A1 = snp.split(':')[1].split('_')[1]
			SNP_check_A2 = snp.split(':')[1].split('_')[2]
			if(float(main_table['beta'][i]) > 0):
				label = 1
			else:
				label = 0
			# check items: null CpG | null SNP | wrong A1 | wrong A2 | different CHR
			if(CpG_CHR != 'space' and SNP_CHR != 'space' and len(SNP_check_A1) == 1 and len(SNP_check_A2) == 1 and str(CpG_CHR) == str(SNP_CHR)):
				new_table = new_table._append([{'CpG':cpg, 'SNP':snp, 'Beta':main_table['beta'][i], 'Ref':SNP_check_A1, 
										'Alt':SNP_check_A2, 'CHR':CpG_CHR, 'CpG_POS':CpG_POS, 'SNP_POS':SNP_POS,'label':label}], ignore_index=True)
		except IndexError:
			continue       
	
	print(new_table.head())
	print(new_table.shape)
	sum += new_table.shape[0]
	new_table.to_pickle(save_path + chr + '.pkl')

print(sum) # 350949
'''
          CpG              SNP       Beta Ref Alt   CHR    CpG_POS    SNP_POS label
0  cg17489312    1:9001090_C_T -13.285425   C   T  chr1    9376039    9001090     0
1  cg17489312    1:9001125_G_A -13.242602   G   A  chr1    9376039    9001125     0
2  cg17642700  1:199448224_T_C  11.451724   T   C  chr1  200333573  199448224     1
3  cg11309897  1:171261193_G_A  14.365513   G   A  chr1  172106298  171261193     1
4  cg11309897  1:171277506_C_T  14.556287   C   T  chr1  172106298  171277506     1
'''