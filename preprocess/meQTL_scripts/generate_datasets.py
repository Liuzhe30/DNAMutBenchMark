# training: chr1-9,chr14-22
# validation: chr12-13
# test: chr10-11

import pandas as pd
pd.set_option('display.max_columns', None)

cd4_path = '/lustre/home/acct-bmelgn/bmelgn-4/methven/datasets/merge_dnabert/'
mono_path = '/lustre/home/acct-bmelgn/bmelgn-4/methven/datasets/merge_dnabert_Mono/'
output_path_1 = '/lustre/home/acct-bmelgn/share/DNAMutBenchMark/ori/'
output_path_cd4_1 = '/lustre/home/acct-bmelgn/share/DNAMutBenchMark/datasets/benchmark_meqtl_dataset/slope_prediction/CD4/'
output_path_mono_1 = '/lustre/home/acct-bmelgn/share/DNAMutBenchMark/datasets/benchmark_meqtl_dataset/slope_prediction/mono/'
output_path_cd4_2 = '/lustre/home/acct-bmelgn/share/DNAMutBenchMark/datasets_embedding/dnabert2/meqtl_datasets/slope_prediction/CD4/'
output_path_mono_2 = '/lustre/home/acct-bmelgn/share/DNAMutBenchMark/datasets_embedding/dnabert2/meqtl_datasets/slope_prediction/mono/'

model_size = ['small','large']

# cd4
for model in model_size:
    print(model)
    all_clean = pd.DataFrame()
    train_data = pd.DataFrame()
    valid_data = pd.DataFrame()
    test_data = pd.DataFrame()
    for i in range(22):
        chr_str = 'chr' + str(i+1)
        chr = i + 1
        slope_data = pd.read_pickle(cd4_path + chr_str + '_' + model + '.dataset')
        if((chr >= 1 and chr <= 9) or (chr >= 14 and chr <= 22)):
            train_data = pd.concat([train_data,slope_data])
        elif(chr >= 12 and chr <= 13):
            valid_data = pd.concat([valid_data,slope_data])
        else:
            test_data = pd.concat([test_data,slope_data])
        all_clean = pd.concat([all_clean,slope_data])
    
    all_clean = all_clean[['CpG', 'SNP', 'Beta', 'Ref', 'Alt', 'CHR', 'CpG_POS','SNP_POS']].reset_index(drop=True)
    all_clean.to_pickle(output_path_1 + 'CD4_' + model + '.dataset')

    train_data1 = train_data[['CpG', 'SNP', 'Beta', 'Ref', 'Alt', 'CHR', 'CpG_POS','SNP_POS','seq_before','seq_after','seq_len']].reset_index(drop=True)
    train_data1.to_pickle(output_path_cd4_1 + model + '_train.dataset')
    valid_data1 = valid_data[['CpG', 'SNP', 'Beta', 'Ref', 'Alt', 'CHR', 'CpG_POS','SNP_POS','seq_before','seq_after','seq_len']].reset_index(drop=True)
    valid_data1.to_pickle(output_path_cd4_1 + model + '_valid.dataset')
    test_data1 = test_data[['CpG', 'SNP', 'Beta', 'Ref', 'Alt', 'CHR', 'CpG_POS','SNP_POS','seq_before','seq_after','seq_len']].reset_index(drop=True)
    test_data1.to_pickle(output_path_cd4_1 + model + '_test.dataset')

    train_data2 = train_data[['CpG', 'SNP', 'Beta', 'Ref', 'Alt', 'CHR', 'CpG_POS','SNP_POS','seq_before','seq_after','seq_len','dnabert_before','dnabert_after']].reset_index(drop=True)
    train_data2.to_pickle(output_path_cd4_2 + model + '_train.dataset')
    valid_data2 = valid_data[['CpG', 'SNP', 'Beta', 'Ref', 'Alt', 'CHR', 'CpG_POS','SNP_POS','seq_before','seq_after','seq_len','dnabert_before','dnabert_after']].reset_index(drop=True)
    valid_data2.to_pickle(output_path_cd4_2 + model + '_valid.dataset')
    test_data2 = test_data[['CpG', 'SNP', 'Beta', 'Ref', 'Alt', 'CHR', 'CpG_POS','SNP_POS','seq_before','seq_after','seq_len','dnabert_before','dnabert_after']].reset_index(drop=True)
    test_data2.to_pickle(output_path_cd4_2 + model + '_test.dataset')

# mono
for model in model_size:
    print(model)
    all_clean = pd.DataFrame()
    train_data = pd.DataFrame()
    valid_data = pd.DataFrame()
    test_data = pd.DataFrame()
    for i in range(22):
        chr_str = 'chr' + str(i+1)
        chr = i + 1
        slope_data = pd.read_pickle(mono_path + chr_str + '_' + model + '.dataset')
        if((chr >= 1 and chr <= 9) or (chr >= 14 and chr <= 22)):
            train_data = pd.concat([train_data,slope_data])
        elif(chr >= 12 and chr <= 13):
            valid_data = pd.concat([valid_data,slope_data])
        else:
            test_data = pd.concat([test_data,slope_data])
        all_clean = pd.concat([all_clean,slope_data])
    all_clean = all_clean[['CpG', 'SNP', 'Beta', 'Ref', 'Alt', 'CHR', 'CpG_POS','SNP_POS']].reset_index(drop=True)
    all_clean.to_pickle(output_path_1 + 'mono_' + model + '.dataset')

    train_data1 = train_data[['CpG', 'SNP', 'Beta', 'Ref', 'Alt', 'CHR', 'CpG_POS','SNP_POS','seq_before','seq_after','seq_len']].reset_index(drop=True)
    train_data1.to_pickle(output_path_mono_1 + model + '_train.dataset')
    valid_data1 = valid_data[['CpG', 'SNP', 'Beta', 'Ref', 'Alt', 'CHR', 'CpG_POS','SNP_POS','seq_before','seq_after','seq_len']].reset_index(drop=True)
    valid_data1.to_pickle(output_path_mono_1 + model + '_valid.dataset')
    test_data1 = test_data[['CpG', 'SNP', 'Beta', 'Ref', 'Alt', 'CHR', 'CpG_POS','SNP_POS','seq_before','seq_after','seq_len']].reset_index(drop=True)
    test_data1.to_pickle(output_path_mono_1 + model + '_test.dataset')

    train_data2 = train_data[['CpG', 'SNP', 'Beta', 'Ref', 'Alt', 'CHR', 'CpG_POS','SNP_POS','seq_before','seq_after','seq_len','dnabert_before','dnabert_after']].reset_index(drop=True)
    train_data2.to_pickle(output_path_mono_2 + model + '_train.dataset')
    valid_data2 = valid_data[['CpG', 'SNP', 'Beta', 'Ref', 'Alt', 'CHR', 'CpG_POS','SNP_POS','seq_before','seq_after','seq_len','dnabert_before','dnabert_after']].reset_index(drop=True)
    valid_data2.to_pickle(output_path_mono_2 + model + '_valid.dataset')
    test_data2 = test_data[['CpG', 'SNP', 'Beta', 'Ref', 'Alt', 'CHR', 'CpG_POS','SNP_POS','seq_before','seq_after','seq_len','dnabert_before','dnabert_after']].reset_index(drop=True)
    test_data2.to_pickle(output_path_mono_2 + model + '_test.dataset')