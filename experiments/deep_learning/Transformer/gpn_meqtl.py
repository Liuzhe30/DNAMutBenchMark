import os
import pandas as pd
import numpy as np
import seaborn as sns
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import warnings

warnings.filterwarnings('ignore')

import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

embedding_name = 'gpn'
task_name = 'slope_prediction'
model_size = ['small']
tissue_list = ['CD4', 'mono']
for repeat_time in ['1', '2', '3']:
    for model_name in model_size:
        for tissue in tissue_list:
            print('Model: ' + model_name + ', Tissue: ' + tissue)
            data_path = '../dataset/' + embedding_name + '/meqtl_datasets/' + task_name + '/' + tissue + '/'
            image_path = 'images/' + embedding_name + '/meqtl/not_shuffled/'

            train_data = pd.read_pickle(data_path + model_name + '_train.dataset')
            valid_data = pd.read_pickle(data_path + model_name + '_valid.dataset')
            test_data = pd.read_pickle(data_path + model_name + '_test.dataset')

            train_data = train_data.sample(frac=1).reset_index(drop=True)
            valid_data = valid_data.sample(frac=1).reset_index(drop=True)
            test_data = test_data.sample(frac=1).reset_index(drop=True)

            train_data = pd.concat([train_data, valid_data]).reset_index(drop=True)  # for machine learning, grid search
            sample_list = []
            y_list = []
            for i in range(train_data.shape[0]):
                feature_list = []
                feature_list += train_data[embedding_name + '_before'][i].flatten().tolist()
                feature_list += train_data[embedding_name + '_after'][i].flatten().tolist()
                sample_list.append(feature_list)
                y_list.append(train_data['Beta'][i])
            X_train = np.array(sample_list)
            Y_train = np.array(y_list)
            sample_list = []
            y_list = []
            for i in range(test_data.shape[0]):
                feature_list = []
                feature_list += test_data[embedding_name + '_before'][i].flatten().tolist()
                feature_list += test_data[embedding_name + '_after'][i].flatten().tolist()
                sample_list.append(feature_list)
                y_list.append(test_data['Beta'][i])
            X_test = np.array(sample_list)
            Y_test = np.array(y_list)


            class TransformerRegressor(nn.Module):
                def __init__(self, input_size):
                    super(TransformerRegressor, self).__init__()
                    d_model = 128  # 可以根据需要调整
                    nhead = 8
                    self.input_fc = nn.Linear(input_size, d_model)
                    self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
                    self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
                    self.fc = nn.Linear(d_model, 1)

                def forward(self, src):
                    src = self.input_fc(src)
                    output = self.transformer_encoder(src)
                    output = output.mean(dim=0)  # 对序列长度维度求平均
                    output = self.fc(output)
                    return output.squeeze()

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if torch.cuda.is_available():
                current_device = torch.cuda.current_device()
                device_name = torch.cuda.get_device_name(current_device)
                print(f"当前使用的CUDA设备: {device_name} (设备ID: {current_device})")
                print(torch.cuda.device_count())
            else:
                print("CUDA设备不可用")

            X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
            Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32).to(device)
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
            Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32).to(device)

            X_train_tensor = X_train_tensor.unsqueeze(0)
            X_test_tensor = X_test_tensor.unsqueeze(0)

            input_size = X_train.shape[1]
            model = TransformerRegressor(input_size=input_size).to(device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.0001)

            num_epochs = 100
            for epoch in range(num_epochs):
                model.train()
                optimizer.zero_grad()
                outputs = model(X_train_tensor)
                loss = criterion(outputs, Y_train_tensor)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                outputs = model(X_test_tensor)
                outputs = outputs.cpu().numpy()
                Y_test_np = Y_test_tensor.cpu().numpy()

            rmse = np.sqrt(mean_squared_error(Y_test_np, outputs))
            r2 = r2_score(Y_test_np, outputs)
            pcc, pcc_p = pearsonr(Y_test_np.flatten(), outputs.flatten())

            print(f'测试集RMSE: {rmse:.4f}')
            print(f'测试集R²: {r2:.4f}')
            print(f'PCC: {pcc:.4f}, P值: {pcc_p:.4f}')

            with open(image_path + model_name + '-performance-' + tissue + '(' + task_name + ')_' + repeat_time + '.txt', 'w') as f:
                f.write(f'测试集RMSE: {rmse:.4f}\n')
                f.write(f'测试集R²: {r2:.4f}\n')
                f.write(f'PCC: {pcc:.4f}, P值: {pcc_p:.4f}\n')

            ax = plt.subplots(figsize=(3,3))
            plt.ylabel("true label")
            plt.xlabel("pred label")
            sns.regplot(x=outputs,y=Y_test_np,x_jitter = 0.15,y_jitter = 0.15,
                        scatter_kws = {'color':'#2E8B57','alpha':0.7,'s':15}, line_kws={"color": "#006400"},truncate=False)
            plt.title(model_name + ' Model (' + task_name + ')\n' + tissue + ', ' + 'Transformer')

            plt.savefig(image_path + model_name + '-transformer-' + tissue + '(' + task_name + ')_' + repeat_time + '.png',dpi=300, bbox_inches = 'tight')

