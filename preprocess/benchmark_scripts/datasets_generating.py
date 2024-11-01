import pandas as pd
import numpy as np
import sklearn

# randomly selection
fasta_path = '../../datasets/reference_genome_hg38/'
output_path = '../../datasets/benchmark_datasets/'
data_path = '../../datasets/eqtl_datasets/sign_prediction/Esophagus_Mucosa_large.dataset'
data = pd.read_pickle(data_path)
print(data.head())
'''
0   ENSG00000228794.8   chr1_841166_A_G_b38         16028  0.255474        184      210  1.305799e-11  0.291182  0.041555      1
1  ENSG00000131591.17  chr1_1091327_C_A_b38        -25034  0.413625        274      340  3.169983e-21  0.223220  0.022056      1
2   ENSG00000269737.2  chr1_1655861_G_A_b38        -16129  0.030414         25       25  1.190742e-40  1.664350  0.108687      1
3   ENSG00000224387.1  chr1_2522989_T_C_b38         29731  0.227901        148      165  2.671812e-15  0.454298  0.054812      1
4   ENSG00000229393.1  chr1_2522989_T_C_b38         28510  0.227901        148      165  5.657427e-18  0.507771  0.055545      1
'''

shuffled = sklearn.utils.shuffle(data).reset_index(drop=True)
print(shuffled.head())

variant_list = []
for i in range(10):
    variant_list.append(shuffled['variant_id'][i])
print(variant_list)

seq = pd.DataFrame(columns=['variant_id','chr','ref','alt','pos','seq_before','seq_after'])
for i in range(10):
    variant_id = shuffled['variant_id'].values[i]
    chr_str = variant_id.split('_')[0]
    position = int(variant_id.split('_')[1])
    before_mutation = variant_id.split('_')[2]
    after_mutation = variant_id.split('_')[3]  
    with open(fasta_path + chr_str + '.fasta') as fa:
        line = fa.readline()
        before_seq = line[position  - 251:position  + 250] # maxlen = 501
        after_seq = line[position  - 251:position - 1] + after_mutation + line[position:position  + 250]
    
    seq = seq._append({'variant_id':variant_id,'chr':chr_str,'ref':before_mutation,'alt':after_mutation,'pos':position,
                        'seq_before':before_seq,'seq_after':after_seq},ignore_index=True)

print(seq.head())
seq.to_csv(output_path + 'mutation_501bp.csv',index=False)
'''
               variant_id    chr ref alt       pos                                         seq_before                                          seq_after
0  chr13_43039322_A_T_b38  chr13   A   T  43039322  CATTAAAGGTAATGTGAGAGTGACTTCAGATTGTTGCTTGAAGGCA...  CATTAAAGGTAATGTGAGAGTGACTTCAGATTGTTGCTTGAAGGCA...
1   chr6_31407340_T_G_b38   chr6   T   G  31407340  atgtcattagtgttttgatagggattgcattgaatctgtagattac...  atgtcattagtgttttgatagggattgcattgaatctgtagattac...
2  chr10_47980440_C_T_b38  chr10   C   T  47980440  gaaatcacaatgaaaattagaaaattatttgaaatgaatgaaaata...  gaaatcacaatgaaaattagaaaattatttgaaatgaatgaaaata...
3   chr17_8106013_A_C_b38  chr17   A   C   8106013  tggatgcagtgatgtggtggtgcatgcctgtagtcccagctaccag...  tggatgcagtgatgtggtggtgcatgcctgtagtcccagctaccag...
4   chr6_37636844_C_T_b38   chr6   C   T  37636844  TTGTAGGGTGGAGTGGGGAAGTTATCAGTCTTCAGACTGCCCAGGC...  TTGTAGGGTGGAGTGGGGAAGTTATCAGTCTTCAGACTGCCCAGGC...
'''