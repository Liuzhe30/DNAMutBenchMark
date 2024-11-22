import pandas as pd
import numpy as np
import json
from tqdm import tqdm

annotation_path = '../../datasets/genome_annotation_hg38/genome_field/'
type_list = ['proximal_enhancer','distal_enhancer','promoter','CTCF','insulator','K4m3']
'''
  chrom  chromStart  chromEnd               type
0  chr1      181251    181601  proximal_enhancer
1  chr1      190865    191071    distal_enhancer
2  chr1      778562    778912           promoter
3  chr1      779086    779355           promoter
4  chr1      779727    780060  proximal_enhancer
'''
data_path = '../../datasets/benchmark_datasets/mutation_501bp.csv'
data = pd.read_csv(data_path)
print(data.head())
'''
               variant_id    chr ref alt       pos                                         seq_before                                          seq_after
0  chr13_43039322_A_T_b38  chr13   A   T  43039322  CATTAAAGGTAATGTGAGAGTGACTTCAGATTGTTGCTTGAAGGCA...  CATTAAAGGTAATGTGAGAGTGACTTCAGATTGTTGCTTGAAGGCA...
1   chr6_31407340_T_G_b38   chr6   T   G  31407340  atgtcattagtgttttgatagggattgcattgaatctgtagattac...  atgtcattagtgttttgatagggattgcattgaatctgtagattac...
2  chr10_47980440_C_T_b38  chr10   C   T  47980440  gaaatcacaatgaaaattagaaaattatttgaaatgaatgaaaata...  gaaatcacaatgaaaattagaaaattatttgaaatgaatgaaaata...
3   chr17_8106013_A_C_b38  chr17   A   C   8106013  tggatgcagtgatgtggtggtgcatgcctgtagtcccagctaccag...  tggatgcagtgatgtggtggtgcatgcctgtagtcccagctaccag...
4   chr6_37636844_C_T_b38   chr6   C   T  37636844  TTGTAGGGTGGAGTGGGGAAGTTATCAGTCTTCAGACTGCCCAGGC...  TTGTAGGGTGGAGTGGGGAAGTTATCAGTCTTCAGACTGCCCAGGC...
'''

annotation_df = pd.DataFrame()
for i in tqdm(range(data.shape[0])):
    chr = data['chr'][i]
    variant_id = data['variant_id'][i]
    position_list = list(range(data['pos'][i]-250,data['pos'][i]+251)) # 501bp
    annotation_list = []
    with open(annotation_path + chr + '_range.json','r',encoding='utf-8') as load_f:
        load_dict = json.load(load_f)
    for position in position_list: # DNA positions
        flag = 0
        for type in type_list:
            for item in load_dict[type]: # many ranges
                if(position >= item[0] and position <= item[1]):
                    flag = 1
                    annotation_list.append(type)
                    print(type)
                    print(variant_id)
                    pass
        if(flag == 0):
            annotation_list.append('null')
    annotation_df = annotation_df._append({'variant_id':variant_id,'annotation':annotation_list},ignore_index=True)
print(annotation_df.head())
'''
               variant_id                                         annotation
0  chr13_43039322_A_T_b38  [null, null, null, null, null, null, null, nul...
1   chr6_31407340_T_G_b38  [null, null, null, null, null, null, null, nul...
2  chr10_47980440_C_T_b38  [null, null, null, null, null, null, null, nul...
3   chr17_8106013_A_C_b38  [null, null, null, null, null, null, null, nul...
4   chr6_37636844_C_T_b38  [null, null, null, null, null, null, null, nul...
'''
annotation_df.to_pickle('../../datasets/benchmark_datasets/annotation_501bp.pkl')