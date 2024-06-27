import os
import pandas as pd
import numpy as np
import subprocess
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# step 1 CpG and SNP annotation to methylation position of meQTLs
#'''
meqtl_path = '../../../datasets/raw/gwama_cis_random_CD4T_n2/gwama_cis_random_CD4T_'
cpg_annotation_path = '../../../datasets/middlefile/clean_epic/'
save_path = '../../../datasets/middlefile/CD4T_meQTL_annotation/'

'''
main_table:
	              SNP         CpG      beta        se       p-value  n_studies effects
0  1:85591599_G_A  cg15715337 -6.415984  0.565678  8.790000e-30          3   ---??
1  1:85592937_C_T  cg15715337 -6.401692  0.565979  1.260000e-29          3   ---??
2  1:85592946_C_T  cg15715337 -6.399971  0.565982  1.300000e-29          3   ---??
3  1:85593116_C_T  cg15715337 -6.393680  0.566083  1.510000e-29          3   ---??
4  1:15404145_A_G  cg25835058 -3.164003  0.266529  1.820000e-32          3   ---??
'''
'''
cpg_table:
       IlmnID   CHR    CpG_pos
0  cg23229610  chr1    6841125
1  cg25458538  chr1  144921929
2  cg04118974  chr1  110359535
3  cg07659892  chr1  234659347
4  cg11993619  chr1  112286575
'''
'''
indexed_cpg:
             CHR    CpG_pos
IlmnID
cg23229610  chr1    6841125
cg25458538  chr1  144921929
cg04118974  chr1  110359535
cg07659892  chr1  234659347
cg11993619  chr1  112286575
'''
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
          CpG             SNP      Beta Ref Alt   CHR   CpG_POS   SNP_POS label
0  cg15715337  1:85591599_G_A -6.415984   G   A  chr1  85600447  85591599     0
1  cg15715337  1:85592937_C_T -6.401692   C   T  chr1  85600447  85592937     0
2  cg15715337  1:85592946_C_T -6.399971   C   T  chr1  85600447  85592946     0
3  cg15715337  1:85593116_C_T -6.393680   C   T  chr1  85600447  85593116     0
4  cg25835058  1:15404145_A_G -3.164003   A   G  chr1  15407757  15404145     0

          CpG              SNP      Beta Ref Alt   CHR    CpG_POS    SNP_POS label
0  cg02317125   7:25298131_C_T -3.376621   C   T  chr7   25303453   25298131     0
1  cg17418956  7:158719084_C_T  3.972300   C   T  chr7  158672758  158719084     1
2  cg17418956  7:158745908_A_C  4.424289   A   C  chr7  158672758  158745908     1
3  cg17418956  7:158747040_G_A  4.543595   G   A  chr7  158672758  158747040     1
4  cg17418956  7:158742451_A_G  4.546464   A   G  chr7  158672758  158742451     1
'''




