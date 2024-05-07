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
            tss_after_seq = line[tss_position - 1001:tss_position + tss_distance - 1] + after_mutation + tss_before_seq[tss_position + tss_distance:tss_position + 1000]
            data.loc[i, 'seq_before'] = tss_before_seq
            data.loc[i, 'seq_after'] = tss_after_seq
    data.to_pickle(output_path + bulk + '/small.dataset')
print(data.head())
'''
         phenotype_id              variant_id  tss_distance       maf  \
0   ENSG00000253764.1    chr8_1975478_G_A_b38           841  0.428082
1   ENSG00000085871.8  chr4_139665547_C_A_b38          -221  0.087900
2  ENSG00000111077.17  chr12_53047831_C_A_b38           780  0.231735
3  ENSG00000105607.12  chr19_12891749_T_G_b38           723  0.436073
4   ENSG00000232450.1  chr1_113699628_C_T_b38            -3  0.279680

  ma_samples ma_count  pval_nominal     slope  slope_se  \
0        295      375  3.219436e-05 -0.191308  0.045446
1         77       77  2.916482e-03 -0.159111  0.053100
2        181      203  1.910919e-13 -0.180075  0.023573
3        297      382  1.863291e-10  0.139372  0.021256
4        212      245  1.193495e-02 -0.163614  0.064756

                                          seq_before  \
0  CCCGAAAAcgccccgtgtgcacgcgcccgcccccctccccgcgccc...
1  cccaggctgaagtgcagtggaacgaccttggctcactgcagtctca...
2  AGTACCAGGCACTGCAAAAGAAGGGAATGAGGTCAGACCATAGAAA...
3  TGAGAATCAAGACACACACATTTCTCAACAGATACACAATCAGAAT...
4  acacacacacacacacacacacacacacatacacacatatataaaa...

                                           seq_after
0  CCCGAAAAcgccccgtgtgcacgcgcccgcccccctccccgcgccc...
1  cccaggctgaagtgcagtggaacgaccttggctcactgcagtctca...
2  AGTACCAGGCACTGCAAAAGAAGGGAATGAGGTCAGACCATAGAAA...
3  TGAGAATCAAGACACACACATTTCTCAACAGATACACAATCAGAAT...
'''

# middle model
for bulk in gtex_list:
    data = pd.read_pickle(file_path + bulk + '_middle.dataset')
    data['seq_before'] = 0
    data['seq_after'] = 0
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
            tss_after_seq = line[tss_position - 10001:tss_position + tss_distance - 1] + after_mutation + tss_before_seq[tss_position + tss_distance:tss_position + 10000]
            data.loc[i, 'seq_before'] = tss_before_seq
            data.loc[i, 'seq_after'] = tss_after_seq
    data.to_pickle(output_path + bulk + '/middle.dataset')

# large model
for bulk in gtex_list:
    data = pd.read_pickle(file_path + bulk + '_large.dataset')
    data['seq_before'] = 0
    data['seq_after'] = 0
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
            tss_after_seq = line[tss_position - 100001:tss_position + tss_distance - 1] + after_mutation + tss_before_seq[tss_position + tss_distance:tss_position + 100000]
            data.loc[i, 'seq_before'] = tss_before_seq
            data.loc[i, 'seq_after'] = tss_after_seq
    data.to_pickle(output_path + bulk + '/large.dataset')