# sampling tissue-specific fine-mapped eQTLs
import pandas as pd
from pandas import read_parquet
pd.set_option('display.max_columns', None)

gtex_list = ['Heart_Left_Ventricle','Esophagus_Mucosa','Nerve_Tibial']

non_causal_path = '../../datasets/eqtl_datasets/raw/ppc_0.001_single_mutation/'
causal_path = '../../datasets/eqtl_datasets/raw/ppc_0.9_single_mutation/'
output_path = '../../datasets/eqtl_datasets/middlefile/1_split_tss_distance/'
output_path2 = '../../datasets/eqtl_datasets/'

# non-causal mutations
for bulk in gtex_list:
    small_final_df = pd.DataFrame()
    middle_final_df = pd.DataFrame()
    large_final_df = pd.DataFrame()
    for chr_no in range(1,23):
        raw_df = pd.read_pickle(non_causal_path + bulk + '_' + str(chr_no) + '.pkl')
        raw_df = raw_df.drop(columns=['level_0','index']).reset_index(drop=True)
        raw_df['tss_distance'] = raw_df['tss_distance'].astype(int)
        raw_df['slope'] = raw_df['slope'].astype(float)
        small_df = raw_df[(raw_df['tss_distance']>-1000)&((raw_df['tss_distance']<1000))].reset_index(drop=True)
        small_final_df = pd.concat([small_final_df,small_df])
        middle_df = raw_df[((raw_df['tss_distance']>-10000)&(raw_df['tss_distance']<-1000))|((raw_df['tss_distance']>1000)&(raw_df['tss_distance']<10000))].reset_index(drop=True)
        middle_final_df = pd.concat([middle_final_df,middle_df])
        large_df = raw_df[((raw_df['tss_distance']>-100000)&(raw_df['tss_distance']<-10000))|((raw_df['tss_distance']>10000)&(raw_df['tss_distance']<100000))].reset_index(drop=True)
        large_final_df = pd.concat([large_final_df,large_df])
    small_final_df = small_final_df[(small_final_df['slope']>=-0.2)&(small_final_df['slope']<=0.2)].reset_index(drop=True)
    small_final_df = small_final_df.sample(frac=1).reset_index(drop=True)[0:100]
    small_final_df.to_pickle(output_path + bulk + '_small_non.dataset')
    print(small_final_df.shape)
    middle_final_df = middle_final_df[(middle_final_df['slope']>=-0.2)&(middle_final_df['slope']<=0.2)].reset_index(drop=True)
    middle_final_df = middle_final_df.sample(frac=1).reset_index(drop=True)[0:100]
    middle_final_df.to_pickle(output_path + bulk + '_middle_non.dataset')
    print(middle_final_df.shape)
    large_final_df = large_final_df[(large_final_df['slope']>=-0.2)&(large_final_df['slope']<=0.2)].reset_index(drop=True)
    large_final_df = large_final_df.sample(frac=1).reset_index(drop=True)[0:100]
    large_final_df.to_pickle(output_path + bulk + '_large_non.dataset')
    print(large_final_df.shape)

# up-regulation mutations
for bulk in gtex_list:
    print(bulk)
    small_final_df = pd.DataFrame()
    middle_final_df = pd.DataFrame()
    large_final_df = pd.DataFrame()
    for chr_no in range(1,23):
        raw_df = pd.read_pickle(causal_path + bulk + '_' + str(chr_no) + '.pkl')
        raw_df = raw_df.drop(columns=['level_0','index']).reset_index(drop=True)
        raw_df['tss_distance'] = raw_df['tss_distance'].astype(int)
        raw_df['slope'] = raw_df['slope'].astype(float)
        small_df = raw_df[(raw_df['tss_distance']>-1000)&((raw_df['tss_distance']<1000))].reset_index(drop=True)
        small_final_df = pd.concat([small_final_df,small_df])
        middle_df = raw_df[((raw_df['tss_distance']>-10000)&(raw_df['tss_distance']<-1000))|((raw_df['tss_distance']>1000)&(raw_df['tss_distance']<10000))].reset_index(drop=True)
        middle_final_df = pd.concat([middle_final_df,middle_df])
        large_df = raw_df[((raw_df['tss_distance']>-100000)&(raw_df['tss_distance']<-10000))|((raw_df['tss_distance']>10000)&(raw_df['tss_distance']<100000))].reset_index(drop=True)
        large_final_df = pd.concat([large_final_df,large_df])
    small_final_df = small_final_df[small_final_df['slope']>0.2].reset_index(drop=True)
    small_final_df.to_pickle(output_path + bulk + '_small_up.dataset')
    print(small_final_df.shape)
    middle_final_df = middle_final_df[middle_final_df['slope']>0.2].reset_index(drop=True)
    middle_final_df.to_pickle(output_path + bulk + '_middle_up.dataset')
    print(middle_final_df.shape)
    large_final_df = large_final_df[large_final_df['slope']>0.2].reset_index(drop=True)
    large_final_df.to_pickle(output_path + bulk + '_large_up.dataset')
    print(large_final_df.shape)

# down-regulation mutations
for bulk in gtex_list:
    print(bulk)
    small_final_df = pd.DataFrame()
    middle_final_df = pd.DataFrame()
    large_final_df = pd.DataFrame()
    for chr_no in range(1,23):
        raw_df = pd.read_pickle(causal_path + bulk + '_' + str(chr_no) + '.pkl')
        raw_df = raw_df.drop(columns=['level_0','index']).reset_index(drop=True)
        raw_df['tss_distance'] = raw_df['tss_distance'].astype(int)
        raw_df['slope'] = raw_df['slope'].astype(float)
        small_df = raw_df[(raw_df['tss_distance']>-1000)&((raw_df['tss_distance']<1000))].reset_index(drop=True)
        small_final_df = pd.concat([small_final_df,small_df])
        middle_df = raw_df[((raw_df['tss_distance']>-10000)&(raw_df['tss_distance']<-1000))|((raw_df['tss_distance']>1000)&(raw_df['tss_distance']<10000))].reset_index(drop=True)
        middle_final_df = pd.concat([middle_final_df,middle_df])
        large_df = raw_df[((raw_df['tss_distance']>-100000)&(raw_df['tss_distance']<-10000))|((raw_df['tss_distance']>10000)&(raw_df['tss_distance']<100000))].reset_index(drop=True)
        large_final_df = pd.concat([large_final_df,large_df])
    small_final_df = small_final_df[small_final_df['slope']<-0.2].reset_index(drop=True)
    small_final_df.to_pickle(output_path + bulk + '_small_down.dataset')
    print(small_final_df.shape)
    middle_final_df = middle_final_df[middle_final_df['slope']<-0.2].reset_index(drop=True)
    middle_final_df.to_pickle(output_path + bulk + '_middle_down.dataset')
    print(middle_final_df.shape)
    large_final_df = large_final_df[large_final_df['slope']<-0.2].reset_index(drop=True)
    large_final_df.to_pickle(output_path + bulk + '_large_down.dataset')
    print(large_final_df.shape)

# merge bulk datasets
label_list = ['non','up','down']
for bulk in gtex_list:
    small_final_df = pd.DataFrame()
    middle_final_df = pd.DataFrame()
    large_final_df = pd.DataFrame()
    for label in label_list:
        data = pd.read_pickle(output_path + bulk + '_small_' + label + '.dataset')
        small_final_df = pd.concat([small_final_df,data])
        data = pd.read_pickle(output_path + bulk + '_middle_' + label + '.dataset')
        middle_final_df = pd.concat([middle_final_df,data])
        data = pd.read_pickle(output_path + bulk + '_large_' + label + '.dataset')
        large_final_df = pd.concat([large_final_df,data])
    small_final_df = small_final_df.reset_index(drop=True)
    small_final_df.to_pickle(output_path2 + bulk + '_small.dataset')
    middle_final_df = middle_final_df.reset_index(drop=True)
    middle_final_df.to_pickle(output_path2 + bulk + '_middle.dataset')
    large_final_df = large_final_df.reset_index(drop=True)
    large_final_df.to_pickle(output_path2 + bulk + '_large.dataset')