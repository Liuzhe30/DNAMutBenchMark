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
from contextlib import redirect_stdout


embedding_name = 'enformer-pca'
task_name = 'sign_prediction'
model_size = ['small', 'middle', 'large']
tissue_list = ['Heart_Left_Ventricle', 'Esophagus_Mucosa', 'Nerve_Tibial']
for repeat_time in ['1', '2', '3']:
    for model_name in model_size:
        for tissue in tissue_list:
            if repeat_time == '1' and model_name in ['middle', 'small']:
                print('ok')
                continue
            print('Model: ' + model_name + ', Tissue: ' + tissue)
            data_path = '../dataset/' + embedding_name + '/eqtl_datasets/' + task_name + '/' + tissue + '/'
            image_path = 'images/' + embedding_name + '/sign/not_shuffled/'
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
                feature_list += train_data['enformer_pca' + '_before'][i].flatten().tolist()
                feature_list += train_data['enformer_pca' + '_after'][i].flatten().tolist()
                sample_list.append(feature_list)
                y_list.append(train_data['label'][i])
            X_train = np.array(sample_list)
            Y_train = np.array(y_list)
            sample_list = []
            y_list = []
            for i in range(test_data.shape[0]):
                feature_list = []
                feature_list += test_data['enformer_pca' + '_before'][i].flatten().tolist()
                feature_list += test_data['enformer_pca' + '_after'][i].flatten().tolist()
                sample_list.append(feature_list)
                y_list.append(test_data['label'][i])
            X_test = np.array(sample_list)
            Y_test = np.array(y_list)


            class TransformerClassifier(nn.Module):
                def __init__(self, input_size, num_classes):
                    super(TransformerClassifier, self).__init__()
                    self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=8)
                    self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
                    self.fc = nn.Linear(input_size, num_classes)

                def forward(self, src):
                    output = self.transformer_encoder(src)
                    output = output.mean(dim=0)  # 对序列长度维度求平均
                    output = self.fc(output)
                    return output



            device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
            if torch.cuda.is_available():
                current_device = torch.cuda.current_device()
                device_name = torch.cuda.get_device_name(current_device)
                print(f"当前使用的CUDA设备: {device_name} (设备ID: {current_device})")
                print(torch.cuda.device_count())
            else:
                print("CUDA设备不可用")
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
            Y_train_tensor = torch.tensor(Y_train, dtype=torch.long).to(device)
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
            Y_test_tensor = torch.tensor(Y_test, dtype=torch.long).to(device)

            input_size = X_train.shape[1]
            num_classes = len(np.unique(Y_train))

            X_train_tensor = X_train_tensor.unsqueeze(0)
            X_test_tensor = X_test_tensor.unsqueeze(0)

            model = TransformerClassifier(input_size=input_size, num_classes=num_classes).to(device)
            criterion = nn.CrossEntropyLoss()
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
                _, predicted = torch.max(outputs.data, 1)
                y_score_pro = nn.functional.softmax(outputs, dim=1).cpu().numpy()
                y_score = predicted.cpu().numpy()
                Y_test_np = Y_test_tensor.cpu().numpy()

            y_one_hot = pd.get_dummies(Y_test_np)
            y_score_one_hot = pd.get_dummies(y_score)

            conf_matrix = confusion_matrix(Y_test_np, y_score)
            with open(image_path + model_name + '_cm-transformer-' + tissue + '(' + task_name + ')_' + repeat_time + '.txt', 'w') as file:
                with redirect_stdout(file):
                    print('epoch: {}'.format(num_epochs))
                    print('Accuracy: {}'.format(accuracy_score(Y_test_np, y_score)))
                    print('Precision: {}'.format(precision_score(Y_test_np, y_score, average='weighted')))
                    print('Recall: {}'.format(recall_score(Y_test_np, y_score, average='weighted')))
                    print('F1-Score: {}'.format(f1_score(Y_test_np, y_score, average='weighted')))
            obj1 = confusion_matrix(Y_test, y_score)
            sum_all1 = obj1[0][0] + obj1[0][1]
            sum_all2 = obj1[1][0] + obj1[1][1]
            new_obj = np.array(
                [
                    [float(obj1[0][0]) / sum_all1, float(obj1[0][1]) / sum_all1],
                    [float(obj1[1][0]) / sum_all2, float(obj1[1][1]) / sum_all2]
                ]
            )
            plt.subplots(figsize=(3, 3))
            sns.heatmap(new_obj, fmt='.2f', cmap='PuBuGn', annot=True)
            plt.xlabel("True Label")
            plt.ylabel("Predicted Label")
            plt.title(model_name + ' Model (' + task_name + ')\n' + tissue + ', ' + 'Transformer')
            plt.savefig(image_path + model_name + '_cm-transformer-' + tissue + '(' + task_name + ')_' + repeat_time + '.png', dpi=300,
                        bbox_inches='tight')

            fpr, tpr, thresholds = roc_curve(y_one_hot.to_numpy().ravel(), y_score_pro.ravel())
            auc_ = auc(fpr, tpr)
            plt.subplots(figsize=(3, 3))
            plt.title(model_name + ' Model (' + task_name + ')\n' + tissue + ', ' + 'Transformer')
            plt.plot(fpr, tpr, linewidth=2, label='AUC=%.3f' % auc_, color='#006400')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.axis([0, 1.1, 0, 1.1])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend(loc="lower right")
            plt.savefig(image_path + model_name + '_auc-transformer-' + tissue + '(' + task_name + ')_' + repeat_time + '.png', dpi=300,
                        bbox_inches='tight')
