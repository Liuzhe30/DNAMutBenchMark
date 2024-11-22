import pandas as pd
import argparse

eqtl_tissues = ['Esophagus_Mucosa','Heart_Left_Ventricle','Nerve_Tibial']
eqtl_model_size = ['small','middle','large']
model_list = ['xgboost','lightgbm','random forest','knn','svm']
seed_list = ['seed17','seed510','seed1030']

def fetch_eqtl_slope_results(embedding):
    input_path = '../experiments/machine_learning/eqtl_prediction/slope_prediction/' + embedding + '/'
    output_path = '../experiments/visualization/downstream_tasks_results/'
    output_df = pd.read_csv(output_path + 'eqtl_slope_ML_' + embedding + '_result.csv')
    output_df['embedding'] = embedding

    # no shuffle
    for tissue in eqtl_tissues:
        for seed in seed_list:
            intseed = int(seed.split('seed')[1])
            file_name = input_path + tissue + '_' + seed + '.ipynb'
            rmse_list = []
            r2_list = []
            pcc_list = []
            with open(file_name) as f:
                line = f.readline()
                while line:
                    if('rmse= ' in line):
                        rmse_list.append(float(line.split('rmse= ')[1][0:5]))
                    if('r2= ' in line):
                        r2_list.append(float(line.split('r2= ')[1][0:5]))
                    if('pcc= ' in line):
                        if(line.split('pcc= ')[1][0:5]=='nan\\n'):
                            pcc_list.append(0)
                        else:
                            pcc_list.append(float(line.split('pcc= ')[1][0:5]))
                    line = f.readline()
            index=0
            for model_size in eqtl_model_size:
                for ml_model in model_list:
                    output_df.loc[(output_df['tissue']==tissue)&(output_df['size']==model_size)&(output_df['model']==ml_model)&(output_df['shuffled']=='no')&(output_df['seed']==intseed),'rmse'] = rmse_list[index]
                    output_df.loc[(output_df['tissue']==tissue)&(output_df['size']==model_size)&(output_df['model']==ml_model)&(output_df['shuffled']=='no')&(output_df['seed']==intseed),'r2'] = r2_list[index]
                    output_df.loc[(output_df['tissue']==tissue)&(output_df['size']==model_size)&(output_df['model']==ml_model)&(output_df['shuffled']=='no')&(output_df['seed']==intseed),'pcc'] = pcc_list[index]
                    index += 1
    
    # shuffled
    for tissue in eqtl_tissues:
        for seed in seed_list:
            intseed = int(seed.split('seed')[1])
            file_name = input_path + tissue + '_shuffled_' + seed + '.ipynb'
            rmse_list = []
            r2_list = []
            pcc_list = []
            with open(file_name) as f:
                line = f.readline()
                while line:
                    if('rmse= ' in line):
                        rmse_list.append(float(line.split('rmse= ')[1][0:5]))
                    if('r2= ' in line):
                        r2_list.append(float(line.split('r2= ')[1][0:5]))
                    if('pcc= ' in line):
                        if(line.split('pcc= ')[1][0:5]=='nan\\n'):
                            pcc_list.append(0)
                        else:
                            pcc_list.append(float(line.split('pcc= ')[1][0:5]))
                    line = f.readline()
            index=0
            for model_size in eqtl_model_size:
                for ml_model in model_list:
                    output_df.loc[(output_df['tissue']==tissue)&(output_df['size']==model_size)&(output_df['model']==ml_model)&(output_df['shuffled']=='yes')&(output_df['seed']==intseed),'rmse'] = rmse_list[index]
                    output_df.loc[(output_df['tissue']==tissue)&(output_df['size']==model_size)&(output_df['model']==ml_model)&(output_df['shuffled']=='yes')&(output_df['seed']==intseed),'r2'] = r2_list[index]
                    output_df.loc[(output_df['tissue']==tissue)&(output_df['size']==model_size)&(output_df['model']==ml_model)&(output_df['shuffled']=='yes')&(output_df['seed']==intseed),'pcc'] = pcc_list[index]
                    index += 1

    output_df.to_csv(output_path + 'eqtl_slope_ML_' + embedding + '_result.csv',index=False)

meqtl_tissues = ['CD4','mono']
meqtl_model_size = ['small']

def fetch_meqtl_slope_results(embedding):
    input_path = '../experiments/machine_learning/meqtl_prediction/slope_prediction/' + embedding + '/'
    output_path = '../experiments/visualization/downstream_tasks_results/'
    output_df = pd.read_csv(output_path + 'meqtl_slope_ML_' + embedding + '_result.csv')
    output_df['embedding'] = embedding

    # no shuffle
    for tissue in meqtl_tissues:
        for seed in seed_list:
            intseed = int(seed.split('seed')[1])
            file_name = input_path + tissue + '_' + seed + '.ipynb'
            rmse_list = []
            r2_list = []
            pcc_list = []
            with open(file_name) as f:
                line = f.readline()
                while line:
                    if('rmse= ' in line):
                        rmse_list.append(float(line.split('rmse= ')[1][0:5]))
                    if('r2= ' in line):
                        r2_list.append(float(line.split('r2= ')[1][0:5]))
                    if('pcc= ' in line):
                        if(line.split('pcc= ')[1][0:5]=='nan\\n'):
                            pcc_list.append(0)
                        else:
                            pcc_list.append(float(line.split('pcc= ')[1][0:5]))
                    line = f.readline()
            index=0
            for model_size in meqtl_model_size:
                for ml_model in model_list:
                    output_df.loc[(output_df['tissue']==tissue)&(output_df['size']==model_size)&(output_df['model']==ml_model)&(output_df['shuffled']=='no')&(output_df['seed']==intseed),'rmse'] = rmse_list[index]
                    output_df.loc[(output_df['tissue']==tissue)&(output_df['size']==model_size)&(output_df['model']==ml_model)&(output_df['shuffled']=='no')&(output_df['seed']==intseed),'r2'] = r2_list[index]
                    output_df.loc[(output_df['tissue']==tissue)&(output_df['size']==model_size)&(output_df['model']==ml_model)&(output_df['shuffled']=='no')&(output_df['seed']==intseed),'pcc'] = pcc_list[index]
                    index += 1
    
    # shuffled
    for tissue in meqtl_tissues:
        for seed in seed_list:
            intseed = int(seed.split('seed')[1])
            file_name = input_path + tissue + '_shuffled_' + seed + '.ipynb'
            rmse_list = []
            r2_list = []
            pcc_list = []
            with open(file_name) as f:
                line = f.readline()
                while line:
                    if('rmse= ' in line):
                        rmse_list.append(float(line.split('rmse= ')[1][0:5]))
                    if('r2= ' in line):
                        r2_list.append(float(line.split('r2= ')[1][0:5]))
                    if('pcc= ' in line):
                        if(line.split('pcc= ')[1][0:5]=='nan\\n'):
                            pcc_list.append(0)
                        else:
                            pcc_list.append(float(line.split('pcc= ')[1][0:5]))
                    line = f.readline()
            index=0
            for model_size in meqtl_model_size:
                for ml_model in model_list:
                    output_df.loc[(output_df['tissue']==tissue)&(output_df['size']==model_size)&(output_df['model']==ml_model)&(output_df['shuffled']=='yes')&(output_df['seed']==intseed),'rmse'] = rmse_list[index]
                    output_df.loc[(output_df['tissue']==tissue)&(output_df['size']==model_size)&(output_df['model']==ml_model)&(output_df['shuffled']=='yes')&(output_df['seed']==intseed),'r2'] = r2_list[index]
                    output_df.loc[(output_df['tissue']==tissue)&(output_df['size']==model_size)&(output_df['model']==ml_model)&(output_df['shuffled']=='yes')&(output_df['seed']==intseed),'pcc'] = pcc_list[index]
                    index += 1

    output_df.to_csv(output_path + 'meqtl_slope_ML_' + embedding + '_result.csv',index=False)

def fetch_eqtl_sign_results(embedding):
    input_path = '../experiments/machine_learning/eqtl_prediction/sign_prediction/' + embedding + '/'
    output_path = '../experiments/visualization/downstream_tasks_results/eqtl_sign_ML_' + embedding + '_result.csv'
    output_df = pd.DataFrame()

    # no shuffle
    for tissue in eqtl_tissues:
        for seed in seed_list:
            intseed = int(seed.split('seed')[1])
            file_name = input_path + tissue + '_' + seed + '.ipynb'
            acc_list = []
            precision_list = []
            recall_list = []
            f1_list = []
            with open(file_name) as f:
                line = f.readline()
                while line:
                    if('accuracy:0' in line or 'accuracy:1' in line):
                        try:
                            acc_list.append(float(line.split('accuracy:')[1][0:4]))
                        except ValueError:
                            acc_list.append(float(line.split('accuracy:')[1][0:3]))
                    if('precision:0' in line or 'precision:1' in line):
                        try:
                            precision_list.append(float(line.split('precision:')[1][0:4]))
                        except ValueError:
                            precision_list.append(float(line.split('precision:')[1][0:3]))
                    if('recall:0' in line or 'recall:1' in line):
                        try:
                            recall_list.append(float(line.split('recall:')[1][0:4]))
                        except ValueError:
                            recall_list.append(float(line.split('recall:')[1][0:3]))
                    if('f1-score:0' in line or 'f1-score:1' in line):
                        try:
                            f1_list.append(float(line.split('f1-score:')[1][0:4]))
                        except ValueError:
                            f1_list.append(float(line.split('f1-score:')[1][0:3]))     
                    line = f.readline()

            index=0
            for model_size in eqtl_model_size:
                for ml_model in model_list:
                    output_df = output_df._append({'size':model_size,'model':ml_model,'embedding':embedding,'tissue':tissue,'seed':intseed,'shuffled':'no',
                    'acc':acc_list[index],'precision':precision_list[index],'recall':recall_list[index],'f1':f1_list[index]},ignore_index=True)
                    index += 1
        
    # shuffled
    for tissue in eqtl_tissues:
        for seed in seed_list:
            intseed = int(seed.split('seed')[1])
            file_name = input_path + tissue + '_shuffled_' + seed + '.ipynb'
            acc_list = []
            precision_list = []
            recall_list = []
            f1_list = []
            with open(file_name) as f:
                line = f.readline()
                while line:
                    if('accuracy:0' in line or 'accuracy:1' in line):
                        try:
                            acc_list.append(float(line.split('accuracy:')[1][0:4]))
                        except ValueError:
                            acc_list.append(float(line.split('accuracy:')[1][0:3]))
                    if('precision:0' in line or 'precision:1' in line):
                        try:
                            precision_list.append(float(line.split('precision:')[1][0:4]))
                        except ValueError:
                            precision_list.append(float(line.split('precision:')[1][0:3]))
                    if('recall:0' in line or 'recall:1' in line):
                        try:
                            recall_list.append(float(line.split('recall:')[1][0:4]))
                        except ValueError:
                            recall_list.append(float(line.split('recall:')[1][0:3]))
                    if('f1-score:0' in line or 'f1-score:1' in line):
                        try:
                            f1_list.append(float(line.split('f1-score:')[1][0:4]))
                        except ValueError:
                            f1_list.append(float(line.split('f1-score:')[1][0:3]))    

                    line = f.readline()

            index=0
            for model_size in eqtl_model_size:
                for ml_model in model_list:
                    #print(tissue,model_size,ml_model,seed,index)
                    output_df = output_df._append({'size':model_size,'model':ml_model,'embedding':embedding,'tissue':tissue,'seed':intseed,'shuffled':'yes',
                    'acc':acc_list[index],'precision':precision_list[index],'recall':recall_list[index],'f1':f1_list[index]},ignore_index=True)
                    index += 1

    output_df.to_csv(output_path,index=False)

if __name__ == "__main__":
        
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--embedding', default='gpn')
    args = parser.parse_args()
    #print(args)    

    embedding = args.embedding

    #fetch_eqtl_slope_results(embedding)
    #fetch_meqtl_slope_results(embedding)
    #fetch_eqtl_sign_results(embedding)