#!/home/lwh/anaconda3/bin/python3.9
import random

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
from model import create_model
from dataLoader_aug import flu_dataloader,cov_dataloader,vir_dataloader,anti_dataloader
import warnings
warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(model, train_loader, val_loader, optimizer, lr, epochs,loss_weight,info,use_moco):
    criterion = nn.CrossEntropyLoss()
    if optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model = model.to(device)

    best_metrics = {
        'epoch': 0,
        'accuracy': 0,
        'precision': 0,
        'recall': 0,
        'f1': 0
    }
    print(info)
    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
            target = target.squeeze()
            data, target = data.to(device), target.to(device)
            if use_moco:
                data = torch.split(data, split_size_or_sections=1, dim=1)
                data1 = torch.squeeze(data[0], dim=1)
                data2 = torch.squeeze(data[1], dim=1)
                output, contrastive_loss = model(data1, data2)
                loss = criterion(output, target)
                loss = loss + loss_weight * contrastive_loss
            else:
                output = model(data)
                loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for data, target in val_loader:
                target = target.squeeze()
                data, target = data.to(device), target.to(device)

                if use_moco:
                    data = torch.split(data, split_size_or_sections=1, dim=1)
                    data = torch.squeeze(data[0], dim=1)
                    output = model(data)
                else:
                    output = model(data)

                preds = torch.argmax(output, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        accuracy = accuracy_score(all_targets, all_preds)
        precision = precision_score(all_targets, all_preds, average='macro')
        recall = recall_score(all_targets, all_preds, average='macro')
        f1 = f1_score(all_targets, all_preds, average='macro')

        if accuracy > best_metrics['accuracy']:
            best_metrics.update({
                'epoch': epoch + 1,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
        if (epoch + 1) % 10 == 0 :
            print(f' Epoch {epoch + 1}/{epochs} - Acc: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')

    with open("results.txt", 'a') as file:
        file.write(f"{info}\n Epoch {best_metrics['epoch']}: Acc: {best_metrics['accuracy']}, Precision: {best_metrics['precision']}, Recall: {best_metrics['recall']}, F1: {best_metrics['f1']}\n")


models = ["CNN","BiLSTM"]
tasks = ["fluhost","covhost","vir","anti"]
# models = ["CNN"]
# tasks = ["vir","anti"]

def traset(lr=0.001,epoch=50,batch=128,loss_weight=0.1, numbers_sites=20 ,use_moco = False,use_attention = True):

    for t in tasks:
        if t == "fluhost":
            segs = ['PB2', 'PB1', 'PA', 'HA', 'NP', 'NA', 'M1', 'NS1']
            slen = {'PB2': 759, 'PB1': 757, 'PA': 716, 'HA': 566,'NP': 498, 'NA': 469,"M1":252,"NS1":230}
            for s in segs:
                train_loader, val_loader = flu_dataloader(s,batch,numbers_sites)
                for m in models:
                    info = t + " " + s + " " + m
                    model = create_model(architecture=m,input_dim=20,output_dim=5,sequence_length=slen[s],use_attention =use_attention,use_moco=use_moco)
                    train_model(model,train_loader, val_loader,"Adam",lr,epoch,loss_weight,info,use_moco)

        elif t == "covhost":
            train_loader, val_loader = cov_dataloader(batch,numbers_sites)
            for m in models:
                info = t + " " + m
                model = create_model(architecture=m, input_dim=20, output_dim=6,sequence_length=2396,use_attention =use_attention,use_moco=use_moco)
                train_model(model, train_loader, val_loader, "Adam", lr, epoch,loss_weight,info,use_moco)

        elif t == "vir":
            segs = ['PB2','PB1', 'PA', 'HA', 'NP', 'NA', 'M1', 'NS1']
            slen = {'PB2': 757, 'PB1': 755, 'PA': 713, 'HA': 583, 'NP': 494, 'NA': 503, "M1": 248, "NS1": 235}
            for s in segs:
                train_loader, val_loader = vir_dataloader(s, batch,numbers_sites)
                for m in models:
                    info = t + " " + s + " " + m
                    model = create_model(architecture=m, input_dim=100, output_dim=2,sequence_length=slen[s],use_attention =use_attention,use_moco=use_moco)
                    train_model(model, train_loader, val_loader, "Adam", lr, epoch,loss_weight,info,use_moco)

        elif t == "anti":
            subtypes = ['H1N1',"H3N2","H5N1"]
            slen = {'H1N1': 325, 'H3N2': 327, 'H5N1': 318}
            for s in subtypes:
                train_loader, val_loader = anti_dataloader(s, batch,numbers_sites)
                for m in models:
                    info = t + " " + s + " " + m
                    model = create_model(architecture=m, input_dim=100, output_dim=2,sequence_length=slen[s],use_attention =use_attention,use_moco=use_moco)
                    train_model(model, train_loader, val_loader, "Adam", lr, epoch,loss_weight,info,use_moco)
if __name__ == '__main__':
    number_of_replacement_sites = [5,10,20,30,40]
    loss_weights = [0.05,0.1,0.3,0.5,0.7,0.9]
    for n in number_of_replacement_sites:
        for w in loss_weights:
            with open("results.txt", 'a') as file:
                file.write(
                    f"===========sites_{n}================loss_weights_{w}==============\n")
            traset(use_attention = False,loss_weight=w,numbers_sites = n,use_moco=True)