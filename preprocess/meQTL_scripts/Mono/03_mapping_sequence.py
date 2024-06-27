# mapping DNA sequences, centered on CpG
from sklearn.utils import shuffle
import pandas as pd
pd.set_option('display.max_columns', None)

fasta_path = '../../../datasets/raw/reference_genome_hg19/'
file_path = '../../../datasets/sampled_Mono/'
output_path = '../../../datasets/seq_mapping_Mono/'

model_size = {'small':10_000,'large':100_000}
for i in range(22):
    chr = 'chr' + str(i+1)
    for model in model_size.keys():
        data = pd.read_pickle(file_path + chr + '_' + model + '.dataset')
        data['seq_before'] = ''
        data['seq_after'] = ''
        data['seq_len'] = 0
        max_range = model_size[model]
        for i in range(len(data)):
            after_mutation = data['Alt'].values[i]
            cpg_pos = int(data['CpG_POS'].values[i])
            snp_pos = int(data['SNP_POS'].values[i])

            with open(fasta_path + chr + '.fasta') as fa:
                line = fa.readline()
                seq_before = line[cpg_pos - max_range - 1:cpg_pos + max_range]
                seq_len = len(seq_before)
                seq_after = line[cpg_pos - max_range - 1:snp_pos - 1] + after_mutation + line[snp_pos:cpg_pos + max_range] # replace A1

            data.loc[i, 'seq_before'] = seq_before
            data.loc[i, 'seq_after'] = seq_after
            data.loc[i, 'seq_len'] = seq_len
            print(data.head())
        data.to_pickle(output_path + chr + '_' + model + '.dataset')

'''
          CpG              SNP       Beta Ref Alt   CHR    CpG_POS    SNP_POS  \
0  cg19757108    1:6602586_T_C   9.846328   T   C  chr1    6605447    6602586
1  cg06900068   1:17421748_A_G   9.260979   A   G  chr1   17426323   17421748
2  cg07573037   1:38450367_A_G  10.139603   A   G  chr1   38447870   38450367
3  cg06900068   1:17428832_C_T   9.476675   C   T  chr1   17426323   17428832
4  cg25006095  1:154678986_C_T  17.361973   C   T  chr1  154680705  154678986

  label  distance                                         seq_before  \
0     1      2861  ttttcctgaaagttaagacagtgctcaaaaaaggatggagccatgG...
1     1      4575  GACCACCGTGACCTCTCTGCCCCAGGCATTGGCTGGCACGTGGTTT...
2     1      2497  cttatacaccaataacagataaacggagagccaaatcatgagtgaa...
3     1      2509  GACCACCGTGACCTCTCTGCCCCAGGCATTGGCTGGCACGTGGTTT...
4     1      1719  AATTTATGGGAAGGAAAGTAAGGGTACAGGGTTTTCAACTTGGGGA...

                                           seq_after  seq_len
0  ttttcctgaaagttaagacagtgctcaaaaaaggatggagccatgG...    20001
1  GACCACCGTGACCTCTCTGCCCCAGGCATTGGCTGGCACGTGGTTT...    20001
2  cttatacaccaataacagataaacggagagccaaatcatgagtgaa...    20001
3  GACCACCGTGACCTCTCTGCCCCAGGCATTGGCTGGCACGTGGTTT...    20001
4  AATTTATGGGAAGGAAAGTAAGGGTACAGGGTTTTCAACTTGGGGA...    20001
'''

# post-process
file_path = '../../../datasets/seq_mapping_Mono/'
output_path = '../../../datasets/seq_mapping_post_Mono/'

for i in range(22):
    chr = 'chr' + str(i+1)
    for model in model_size.keys():
        data = pd.read_pickle(file_path + chr + '_' + model + '.dataset')
        data_check = pd.read_pickle(file_path + chr + '_' + model + '.dataset')
        max_range = model_size[model]
        for i in range(len(data_check)):
            seq_len = data_check['seq_len'][i]
            cpg = data_check['CpG'][i]
            snp = data_check['SNP'][i]
            if(seq_len != max_range * 2 + 1):
                data = data.drop(data[(data['CpG']==cpg)&(data['SNP']==snp)].index)
        data = data.reset_index(drop=True)
        data.to_pickle(output_path + chr + '_' + model + '.dataset')