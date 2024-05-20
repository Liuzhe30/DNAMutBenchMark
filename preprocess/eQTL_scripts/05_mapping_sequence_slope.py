# mapping sequences centered on TSS
import pandas as pd
from pandas import read_parquet
from tqdm import tqdm
pd.set_option('display.max_columns', None)

fasta_path = '../../datasets/reference_genome_hg38/'
file_path = '../../datasets/eqtl_datasets/slope_prediction/'
output_path = '../../datasets/eqtl_datasets/middlefile/2_mapping_sequence/slope_prediction/'

gtex_list = ['Heart_Left_Ventricle','Esophagus_Mucosa','Nerve_Tibial']
model_size = ['small','middle','large']

# small model
for bulk in gtex_list:
    data = pd.read_pickle(file_path + bulk + '_small.dataset')
    data['seq_before'] = 0
    data['seq_after'] = 0
    data['seq_len'] = 0
    for i in tqdm(range(len(data))): 
        variant_id = data['variant_id'].values[i]
        chr_str = variant_id.split('_')[0]
        position = int(variant_id.split('_')[1])
        before_mutation = variant_id.split('_')[2]
        after_mutation = variant_id.split('_')[3]  
        tss_distance = int(data['tss_distance'].values[i])
        tss_position = position - tss_distance

        with open(fasta_path + chr_str + '.fasta') as fa:
            line = fa.readline()
            tss_before_seq = line[tss_position - 1001:tss_position + 1000] # maxlen = 2,001
            if(len(tss_before_seq) != 2001):
                print(variant_id)
            tss_after_seq = line[tss_position - 1001:position - 1] + after_mutation + line[position:tss_position + 1000]
            data.loc[i, 'seq_before'] = tss_before_seq
            data.loc[i, 'seq_after'] = tss_after_seq
            data.loc[i, 'seq_len'] = len(tss_before_seq)
    data.to_pickle(output_path + bulk + '/small.dataset')
print(data.head())
'''
        phenotype_id              variant_id  tss_distance       maf  \
0  ENSG00000013016.15   chr2_31235324_G_A_b38           987  0.331804
1  ENSG00000150756.13   chr5_10245029_A_G_b38           529  0.295107
2  ENSG00000198093.10  chr19_51905229_T_C_b38           189  0.250765
3  ENSG00000140983.13    chr16_667544_G_C_b38          -542  0.137500
4   ENSG00000184441.4  chr21_44330365_G_A_b38          -869  0.429664

  ma_samples ma_count  pval_nominal     slope  slope_se  \
0        181      217      0.002583 -0.125359  0.041215
1        161      193      0.000085  0.186622  0.046767
2        147      164      0.017496  0.113791  0.047597
3         81       88      0.000039 -0.130446  0.031188
4        220      281      0.000002  0.157503  0.032432

                                          seq_before  \
0  gcaaatactccttacgctcctagaacaatggatgtcaattcatcat...
1  ctgtttagtgcagcaaccctcatcaacaatcttagctagatcttct...
2  atgttccaactgcgttcaaataaggcaaacgccgaatggtaaccaa...
3  GGGCTCCTCGTCTCTGGGGTGGGGTGAGGACATCTGCCCTAGAGAG...
4  GGCCTGCAGCCGGCTGCCCACAGTCTGCTGCACGGCCTCCAGCCCC...

                                           seq_after  seq_len
0  gcaaatactccttacgctcctagaacaatggatgtcaattcatcat...     2001
1  ctgtttagtgcagcaaccctcatcaacaatcttagctagatcttct...     2001
2  atgttccaactgcgttcaaataaggcaaacgccgaatggtaaccaa...     2001
3  GGGCTCCTCGTCTCTGGGGTGGGGTGAGGACATCTGCCCTAGAGAG...     2001
4  GGCCTGCAGCCGGCTGCCCACAGTCTGCTGCACGGCCTCCAGCCCC...     2001
'''

# middle model
for bulk in gtex_list:
    data = pd.read_pickle(file_path + bulk + '_middle.dataset')
    data['seq_before'] = 0
    data['seq_after'] = 0
    data['seq_len'] = 0
    for i in tqdm(range(len(data))): 
        variant_id = data['variant_id'].values[i]
        chr_str = variant_id.split('_')[0]
        position = int(variant_id.split('_')[1])
        before_mutation = variant_id.split('_')[2]
        after_mutation = variant_id.split('_')[3]  
        tss_distance = int(data['tss_distance'].values[i])
        tss_position = position - tss_distance

        with open(fasta_path + chr_str + '.fasta') as fa:
            line = fa.readline()
            tss_before_seq = line[tss_position - 10001:tss_position + 10000] # maxlen = 20,001
            if(len(tss_before_seq) != 20001):
                print(variant_id)
            tss_after_seq = line[tss_position - 10001:position - 1] + after_mutation + line[position:tss_position + 10000]
            data.loc[i, 'seq_before'] = tss_before_seq
            data.loc[i, 'seq_after'] = tss_after_seq
            data.loc[i, 'seq_len'] = len(tss_before_seq)
    data.to_pickle(output_path + bulk + '/middle.dataset')

# large model
for bulk in gtex_list:
    data = pd.read_pickle(file_path + bulk + '_large.dataset')
    data['seq_before'] = 0
    data['seq_after'] = 0
    data['seq_len'] = 0
    for i in tqdm(range(len(data))): 
        variant_id = data['variant_id'].values[i]
        chr_str = variant_id.split('_')[0]
        position = int(variant_id.split('_')[1])
        before_mutation = variant_id.split('_')[2]
        after_mutation = variant_id.split('_')[3]  
        tss_distance = int(data['tss_distance'].values[i])
        tss_position = position - tss_distance

        with open(fasta_path + chr_str + '.fasta') as fa:
            line = fa.readline()
            tss_before_seq = line[tss_position - 100001:tss_position + 100000] # maxlen = 200,001
            if(len(tss_before_seq) != 200001):
                print(variant_id)
            tss_after_seq = line[tss_position - 100001:position - 1] + after_mutation + line[position:tss_position + 100000]
            data.loc[i, 'seq_before'] = tss_before_seq
            data.loc[i, 'seq_after'] = tss_after_seq
            data.loc[i, 'seq_len'] = len(tss_before_seq)
    data.to_pickle(output_path + bulk + '/large.dataset')