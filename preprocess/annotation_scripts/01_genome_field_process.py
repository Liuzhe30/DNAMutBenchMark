import pandas as pd

file_path = '../../datasets/genome_annotation_hg38/genome_all_field.csv'
output_path = '../../datasets/genome_annotation_hg38/genome_field/'
data = pd.read_csv(file_path,sep='\t')
print(data.head())
'''
  #chrom  chromStart  chromEnd          name  score strand  thickStart  thickEnd   reserved             ccre encodeLabel    zScore ucscLabel accessionLabel                                    description
0   chr1    181251.0  181601.0  EH38E1310153  488.0      .    181251.0  181601.0  255,167,0  pELS,CTCF-bound        pELS  4.884033      enhP       E1310153  EH38E1310153 proximal enhancer-like signature
1   chr1    190865.0  191071.0  EH38E1310154  179.0      .    190865.0  191071.0  255,205,0  dELS,CTCF-bound        dELS  1.792822      enhD       E1310154    EH38E1310154 distal enhancer-like signature
2   chr1    778562.0  778912.0  EH38E1310158  759.0      .    778562.0  778912.0    255,0,0   PLS,CTCF-bound         PLS  7.598523      prom       E1310158           EH38E1310158 promoter-like signature
3   chr1    779086.0  779355.0  EH38E1310159  304.0      .    779086.0  779355.0    255,0,0   PLS,CTCF-bound         PLS  3.046639      prom       E1310159           EH38E1310159 promoter-like signature
4   chr1    779727.0  780060.0  EH38E1310160  281.0      .    779727.0  780060.0  255,167,0  pELS,CTCF-bound        pELS  2.817434      enhP       E1310160  EH38E1310160 proximal enhancer-like signature
'''
print(data['ucscLabel'].unique()) # ['enhP' 'enhD' 'prom' 'CTCF' 'K4m3']

# select function regions
type_dict = {
    'enhP':'proximal_enhancer',
    'enhD':'distal_enhancer',
    'prom':'promoter',
    'CTCF':'CTCF',
    'K4m3':'K4m3'
}

# filter annotations
chr_list = ['chr'+str(i) for i in range(1,23)]
print(chr_list)
for c in chr_list:
    chr_data = data[data['#chrom'] == c]
    chr_data = chr_data.reset_index(drop=True)
    print(chr_data.head())
    new_chr_data = pd.DataFrame()
    for i in range(chr_data.shape[0]):
        type = chr_data['ucscLabel'][i]
        if(type in type_dict.keys()):
            new_chr_data = new_chr_data._append({'chrom':chr_data['#chrom'][i],'chromStart':chr_data['chromStart'][i],
                                                    'chromEnd':chr_data['chromEnd'][i],'type':type_dict[type]},ignore_index=True)
    print(new_chr_data.head())
    new_chr_data.to_pickle(output_path + c + '_map.pkl')

'''
  chrom  chromStart  chromEnd               type
0  chr1      181251    181601  proximal_enhancer
1  chr1      190865    191071    distal_enhancer
2  chr1      778562    778912           promoter
3  chr1      779086    779355           promoter
4  chr1      779727    780060  proximal_enhancer
'''

# get_annotation(chr):
import json
type_list = ['proximal_enhancer','distal_enhancer','promoter','CTCF','insulator','K4m3']
for i in range(22):
    chr = 'chr' + str(i+1)
    annotation_file = pd.read_pickle(output_path + chr + '_map.pkl')
    annotation_dict = {}

    for type in type_list:
        annotation_dict[type] = []
    for type in type_list:
        filtered = annotation_file[annotation_file['type']==type].reset_index(drop=True)
        for j in range(filtered.shape[0]):
            annotation_dict[type].append([int(filtered['chromStart'][j]),int(filtered['chromEnd'][j])])
    
    with open(output_path + chr + '_range.json','w') as f:
        json.dump(annotation_dict,f,indent=4)