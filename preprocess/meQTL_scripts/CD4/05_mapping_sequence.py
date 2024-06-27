# mapping DNA sequences, centered on CpG
from sklearn.utils import shuffle
import pandas as pd
pd.set_option('display.max_columns', None)

fasta_path = '../../../datasets/raw/reference_genome_hg19/'
file_path = '../../../datasets/sampled/'
output_path = '../../../datasets/seq_mapping/'

'''
         CpG             SNP      Beta Ref Alt   CHR   CpG_POS   SNP_POS label  distance
0  cg15715337  1:85591599_G_A -6.415984   G   A  chr1  85600447  85591599     0      8848
1  cg15715337  1:85592937_C_T -6.401692   C   T  chr1  85600447  85592937     0      7510
2  cg15715337  1:85592946_C_T -6.399971   C   T  chr1  85600447  85592946     0      7501
3  cg15715337  1:85593116_C_T -6.393680   C   T  chr1  85600447  85593116     0      7331
4  cg25835058  1:15404145_A_G -3.164003   A   G  chr1  15407757  15404145     0      3612
'''

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
          CpG              SNP      Beta Ref Alt   CHR    CpG_POS    SNP_POS  \
0  cg18621232   1:31684013_G_A  3.562327   G   A  chr1   31681696   31684013
1  cg00112952  1:247619116_C_T  3.242985   C   T  chr1  247616523  247619116
2  cg03985415  1:229344237_C_T  3.010399   C   T  chr1  229340974  229344237
3  cg14085523  1:207169420_A_G  5.799904   A   G  chr1  207176705  207169420
4  cg10977392  1:157741237_T_A  2.287828   T   A  chr1  157740155  157741237

  label  distance                                         seq_before  \
0     1      2317  CGCCAACCCCAGGAGCCCACACAAGGTGGGGAGTGAGGGACAAGGG...
1     1      2593  gcacctggccTTAtttatttatttatttaggcagggtctcactctg...
2     1      3263  actcttacaacccaataattaaaaagatgaacaatccaattaaata...
3     1      7285  tttttaaatagtgacatatttgggagttttaaaaaagacctctaga...
4     1      1082  gtaatgccagcactttgcgaggccgaggcgggtggattacttgagg...

                                           seq_after  seq_len
0  CGCCAACCCCAGGAGCCCACACAAGGTGGGGAGTGAGGGACAAGGG...    20001
1  gcacctggccTTAtttatttatttatttaggcagggtctcactctg...    20001
2  actcttacaacccaataattaaaaagatgaacaatccaattaaata...    20001
3  tttttaaatagtgacatatttgggagttttaaaaaagacctctaga...    20001
4  gtaatgccagcactttgcgaggccgaggcgggtggattacttgagg...    20001
'''

# post-process
file_path = '../../../datasets/seq_mapping/'
output_path = '../../../datasets/seq_mapping_post/'

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
    
# large-dataset down-sampling
model = 'large'
for i in range(22):
    chr = 'chr' + str(i+1)
    data = pd.read_pickle(output_path + chr + '_' + model + '.dataset')
    data = shuffle(data)
    sample_num = data.shape[0]
    new_data = data.sample(n=int(sample_num/2),replace=False).reset_index(drop=True)
    new_data.to_pickle(output_path + chr + '_' + model + '.dataset')