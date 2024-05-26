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
            #print(data.head())
    data.to_pickle(output_path + bulk + '/small.dataset')
print(data.head())
'''
        phenotype_id             variant_id  tss_distance       maf  \
0   ENSG00000236423.5   chr1_3900688_T_C_b38           316  0.084098
1   ENSG00000090432.6  chr1_20508117_C_A_b38           -44  0.134969
2   ENSG00000228172.5  chr1_25820023_G_C_b38          -738  0.463303
3  ENSG00000117640.17  chr1_25820023_G_C_b38          -775  0.463303
4  ENSG00000000938.12  chr1_27634281_G_A_b38          -996  0.067278

  ma_samples ma_count  pval_nominal     slope  slope_se  \
0         52       55  1.657173e-06  0.342601  0.069935
1         82       88  6.482217e-12  0.276024  0.038419
2        242      303  4.628220e-23  0.513832  0.047313
3        242      303  9.349158e-65  0.530491  0.023360
4         42       44  2.717342e-12  0.523637  0.071486

                                          seq_before  \
0  AGGAGAGCCTCCATGCAGCTCAGAGCCTCCCAAGTGGACCGGGACC...
1  agcccagatcccgccactgcactccagcctgggcgacacagcaaga...
2  CCCGCGGGGGCACGGTCTCGATGGAGGGGAGTGTGCTCCGCGGTAT...
3  CCGCGGTATCGGAGCCTACAGCCGCCAGCGCCTCGCCCACTCGGGG...
4  CGGGGAGCGCGGGCCGAGACCGCCGCGGGCGCGGAGGGGGCGCCCG...

                                           seq_after  seq_len
0  AGGAGAGCCTCCATGCAGCTCAGAGCCTCCCAAGTGGACCGGGACC...     2001
1  agcccagatcccgccactgcactccagcctgggcgacacagcaaga...     2001
2  CCCGCGGGGGCACGGTCTCGATGGAGGGGAGTGTGCTCCGCGGTAT...     2001
3  CCGCGGTATCGGAGCCTACAGCCGCCAGCGCCTCGCCCACTCGGGG...     2001
4  CGGGAAGCGCGGGCCGAGACCGCCGCGGGCGCGGAGGGGGCGCCCG...     2001
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